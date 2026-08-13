#include "cyton_trac_ik_kinematics_plugin/trac_ik_kinematics_plugin.hpp"

#include <kdl_parser/kdl_parser.hpp>

#include <moveit/robot_model/joint_model_group.hpp>
#include <moveit/robot_model/robot_model.hpp>

#include <rclcpp/logging.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace cyton_trac_ik_kinematics_plugin
{

namespace
{
rclcpp::Logger logger()
{
  return rclcpp::get_logger("cyton_trac_ik_kinematics_plugin");
}

geometry_msgs::msg::Pose kdlFrameToPoseMsg(const KDL::Frame& frame)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = frame.p.x();
  pose.position.y = frame.p.y();
  pose.position.z = frame.p.z();
  double qx, qy, qz, qw;
  frame.M.GetQuaternion(qx, qy, qz, qw);
  pose.orientation.x = qx;
  pose.orientation.y = qy;
  pose.orientation.z = qz;
  pose.orientation.w = qw;
  return pose;
}

KDL::Frame poseMsgToKdlFrame(const geometry_msgs::msg::Pose& pose)
{
  return KDL::Frame(
      KDL::Rotation::Quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
      KDL::Vector(pose.position.x, pose.position.y, pose.position.z));
}
}  // namespace

bool TracIKKinematicsPlugin::initialize(const rclcpp::Node::SharedPtr& node,
                                         const moveit::core::RobotModel& robot_model,
                                         const std::string& group_name, const std::string& base_frame,
                                         const std::vector<std::string>& tip_frames, double search_discretization)
{
  storeValues(robot_model, group_name, base_frame, tip_frames, search_discretization);

  if (tip_frames.size() != 1)
  {
    RCLCPP_ERROR(logger(), "TracIKKinematicsPlugin only supports a single tip frame, got %zu",
                 tip_frames.size());
    return false;
  }

  if (!robot_model.getURDF())
  {
    RCLCPP_ERROR(logger(), "RobotModel has no parsed URDF -- cannot build a KDL::Chain from it.");
    return false;
  }

  KDL::Tree tree;
  if (!kdl_parser::treeFromUrdfModel(*robot_model.getURDF(), tree))
  {
    RCLCPP_ERROR(logger(), "kdl_parser::treeFromUrdfModel() failed to build a KDL::Tree from the URDF.");
    return false;
  }

  if (!tree.getChain(base_frame, tip_frames[0], kdl_chain_))
  {
    RCLCPP_ERROR(logger(), "Could not extract a KDL::Chain from '%s' to '%s'.", base_frame.c_str(),
                 tip_frames[0].c_str());
    return false;
  }

  // Collect this chain's actual moving joints (skip fixed joints) and the
  // link name at the end of every segment -- same two lists
  // KDLKinematicsPlugin exposes via getJointNames()/getLinkNames().
  joint_names_.clear();
  link_names_.clear();
  for (unsigned int i = 0; i < kdl_chain_.getNrOfSegments(); ++i)
  {
    const KDL::Segment& segment = kdl_chain_.getSegment(i);
    link_names_.push_back(segment.getName());
    if (segment.getJoint().getType() != KDL::Joint::None)
    {
      joint_names_.push_back(segment.getJoint().getName());
    }
  }

  const unsigned int numJoints = static_cast<unsigned int>(joint_names_.size());
  joint_min_.resize(numJoints);
  joint_max_.resize(numJoints);

  const moveit::core::JointModelGroup* jmg = robot_model.getJointModelGroup(group_name);
  if (!jmg)
  {
    RCLCPP_ERROR(logger(), "RobotModel has no JointModelGroup named '%s'.", group_name.c_str());
    return false;
  }

  for (unsigned int i = 0; i < numJoints; ++i)
  {
    const moveit::core::JointModel* joint_model = robot_model.getJointModel(joint_names_[i]);
    if (!joint_model)
    {
      RCLCPP_ERROR(logger(), "JointModel '%s' (from the KDL chain) not found in the RobotModel.",
                   joint_names_[i].c_str());
      return false;
    }
    const moveit::core::VariableBounds& bounds = joint_model->getVariableBounds(joint_names_[i]);
    joint_min_(i) = bounds.min_position_;
    joint_max_(i) = bounds.max_position_;
  }

  // eps=1e-5 and SolveType::Speed match this project's KDL plugin's own
  // general-purpose defaults as closely as TRAC-IK's API allows -- Speed
  // returns the first valid solution found (fastest), as opposed to
  // Distance (closest to seed) / Manip1 / Manip2 (manipulability-weighted).
  // default_timeout_solver_ matches kinematics.yaml's existing
  // kinematics_solver_timeout value used for the KDL plugin, for a like-
  // for-like comparison between the two solvers.
  trac_ik_solver_ = std::make_unique<TRAC_IK::TRAC_IK>(kdl_chain_, joint_min_, joint_max_,
                                                         default_timeout_solver_, 1e-5, TRAC_IK::Speed);

  fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);

  initialized_ = true;
  RCLCPP_INFO(logger(), "TracIKKinematicsPlugin initialized for group '%s' (%u joints, base='%s', tip='%s').",
              group_name.c_str(), numJoints, base_frame.c_str(), tip_frames[0].c_str());
  return true;
}

bool TracIKKinematicsPlugin::searchPositionIKImpl(const geometry_msgs::msg::Pose& ik_pose,
                                                   const std::vector<double>& ik_seed_state, double timeout,
                                                   const std::vector<double>& consistency_limits,
                                                   std::vector<double>& solution,
                                                   const IKCallbackFn& solution_callback,
                                                   moveit_msgs::msg::MoveItErrorCodes& error_code) const
{
  (void)consistency_limits;  // TRAC-IK has no native consistency-limit concept; not enforced here.
  (void)timeout;             // per-call timeout not threaded into the solver -- see header comment.

  if (!initialized_ || !trac_ik_solver_)
  {
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    return false;
  }

  const unsigned int numJoints = static_cast<unsigned int>(joint_names_.size());
  if (ik_seed_state.size() != numJoints)
  {
    RCLCPP_ERROR(logger(), "Seed state has %zu values, expected %u.", ik_seed_state.size(), numJoints);
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;
    return false;
  }

  KDL::JntArray q_init(numJoints);
  for (unsigned int i = 0; i < numJoints; ++i)
  {
    q_init(i) = ik_seed_state[i];
  }

  const KDL::Frame p_in = poseMsgToKdlFrame(ik_pose);
  KDL::JntArray q_out(numJoints);

  const int result = trac_ik_solver_->CartToJnt(q_init, p_in, q_out);
  if (result <= 0)
  {
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
    return false;
  }

  solution.resize(numJoints);
  for (unsigned int i = 0; i < numJoints; ++i)
  {
    solution[i] = q_out(i);
  }

  if (solution_callback)
  {
    solution_callback(ik_pose, solution, error_code);
    if (error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS)
    {
      return false;
    }
  }
  else
  {
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  }

  return true;
}

bool TracIKKinematicsPlugin::getPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                            const std::vector<double>& ik_seed_state,
                                            std::vector<double>& solution,
                                            moveit_msgs::msg::MoveItErrorCodes& error_code,
                                            const kinematics::KinematicsQueryOptions& /*options*/) const
{
  return searchPositionIKImpl(ik_pose, ik_seed_state, default_timeout_solver_, {}, solution, IKCallbackFn(),
                              error_code);
}

bool TracIKKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                               const std::vector<double>& ik_seed_state, double timeout,
                                               std::vector<double>& solution,
                                               moveit_msgs::msg::MoveItErrorCodes& error_code,
                                               const kinematics::KinematicsQueryOptions& /*options*/) const
{
  return searchPositionIKImpl(ik_pose, ik_seed_state, timeout, {}, solution, IKCallbackFn(), error_code);
}

bool TracIKKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                               const std::vector<double>& ik_seed_state, double timeout,
                                               const std::vector<double>& consistency_limits,
                                               std::vector<double>& solution,
                                               moveit_msgs::msg::MoveItErrorCodes& error_code,
                                               const kinematics::KinematicsQueryOptions& /*options*/) const
{
  return searchPositionIKImpl(ik_pose, ik_seed_state, timeout, consistency_limits, solution, IKCallbackFn(),
                              error_code);
}

bool TracIKKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                               const std::vector<double>& ik_seed_state, double timeout,
                                               std::vector<double>& solution,
                                               const IKCallbackFn& solution_callback,
                                               moveit_msgs::msg::MoveItErrorCodes& error_code,
                                               const kinematics::KinematicsQueryOptions& /*options*/) const
{
  return searchPositionIKImpl(ik_pose, ik_seed_state, timeout, {}, solution, solution_callback, error_code);
}

bool TracIKKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                               const std::vector<double>& ik_seed_state, double timeout,
                                               const std::vector<double>& consistency_limits,
                                               std::vector<double>& solution,
                                               const IKCallbackFn& solution_callback,
                                               moveit_msgs::msg::MoveItErrorCodes& error_code,
                                               const kinematics::KinematicsQueryOptions& /*options*/) const
{
  return searchPositionIKImpl(ik_pose, ik_seed_state, timeout, consistency_limits, solution, solution_callback,
                              error_code);
}

bool TracIKKinematicsPlugin::getPositionFK(const std::vector<std::string>& link_names,
                                            const std::vector<double>& joint_angles,
                                            std::vector<geometry_msgs::msg::Pose>& poses) const
{
  if (!initialized_ || !fk_solver_)
  {
    return false;
  }

  const unsigned int numJoints = static_cast<unsigned int>(joint_names_.size());
  if (joint_angles.size() != numJoints)
  {
    RCLCPP_ERROR(logger(), "getPositionFK: joint_angles has %zu values, expected %u.", joint_angles.size(),
                 numJoints);
    return false;
  }

  KDL::JntArray q(numJoints);
  for (unsigned int i = 0; i < numJoints; ++i)
  {
    q(i) = joint_angles[i];
  }

  poses.resize(link_names.size());
  for (std::size_t i = 0; i < link_names.size(); ++i)
  {
    // Find which chain segment ends at this link, so FK can be computed up
    // to exactly that segment (not just the final tip) -- JntToCart's
    // segmentNr is inclusive of that segment's own transform.
    int segmentIndex = -1;
    for (unsigned int s = 0; s < kdl_chain_.getNrOfSegments(); ++s)
    {
      if (kdl_chain_.getSegment(s).getName() == link_names[i])
      {
        segmentIndex = static_cast<int>(s) + 1;  // JntToCart's segmentNr is 1-indexed (0 = base frame)
        break;
      }
    }
    if (segmentIndex < 0)
    {
      RCLCPP_ERROR(logger(), "getPositionFK: '%s' is not a link in this chain.", link_names[i].c_str());
      return false;
    }

    KDL::Frame frame;
    if (fk_solver_->JntToCart(q, frame, segmentIndex) < 0)
    {
      RCLCPP_ERROR(logger(), "getPositionFK: JntToCart failed for link '%s'.", link_names[i].c_str());
      return false;
    }
    poses[i] = kdlFrameToPoseMsg(frame);
  }

  return true;
}

const std::vector<std::string>& TracIKKinematicsPlugin::getJointNames() const
{
  return joint_names_;
}

const std::vector<std::string>& TracIKKinematicsPlugin::getLinkNames() const
{
  return link_names_;
}

}  // namespace cyton_trac_ik_kinematics_plugin

PLUGINLIB_EXPORT_CLASS(cyton_trac_ik_kinematics_plugin::TracIKKinematicsPlugin, kinematics::KinematicsBase)
