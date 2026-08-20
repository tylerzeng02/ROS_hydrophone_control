#include "pose_dependent_correction.h"

#include <cmath>

namespace pose_dependent_correction {
namespace {

struct Vec3 { double x = 0.0, y = 0.0, z = 0.0; };
struct Mat3 { double m[3][3]; };

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 add(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

Vec3 normalized(const Vec3& v) {
    const double n = std::sqrt(dot(v, v));
    return {v.x / n, v.y / n, v.z / n};
}

Vec3 matVec(const Mat3& R, const Vec3& v) {
    return {
        R.m[0][0] * v.x + R.m[0][1] * v.y + R.m[0][2] * v.z,
        R.m[1][0] * v.x + R.m[1][1] * v.y + R.m[1][2] * v.z,
        R.m[2][0] * v.x + R.m[2][1] * v.y + R.m[2][2] * v.z,
    };
}

Mat3 matMul(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += A.m[i][k] * B.m[k][j];
            }
            C.m[i][j] = s;
        }
    }
    return C;
}

Mat3 identity3() {
    Mat3 I{};
    I.m[0][0] = I.m[1][1] = I.m[2][2] = 1.0;
    return I;
}

// Rodrigues' rotation formula: rotate by `angle` radians about a UNIT axis.
// Matches scipy's Rotation.from_rotvec(axis * angle).as_matrix() used by
// the Python model this ports (see this file's header comment).
Mat3 axisAngleToMatrix(const Vec3& axis, double angle) {
    const double c = std::cos(angle), s = std::sin(angle), t = 1.0 - c;
    const double x = axis.x, y = axis.y, z = axis.z;
    Mat3 R{};
    R.m[0][0] = t * x * x + c;       R.m[0][1] = t * x * y - s * z;   R.m[0][2] = t * x * z + s * y;
    R.m[1][0] = t * x * y + s * z;   R.m[1][1] = t * y * y + c;       R.m[1][2] = t * y * z - s * x;
    R.m[2][0] = t * x * z - s * y;   R.m[2][1] = t * y * z + s * x;   R.m[2][2] = t * z * z + c;
    return R;
}

constexpr int N = 7;
constexpr int SHOULDER_ROLL = 0, SHOULDER_PITCH = 1, SHOULDER_YAW = 2;
constexpr int ELBOW_PITCH = 3, ELBOW_YAW = 4, WRIST_PITCH = 5, WRIST_ROLL = 6;

// Deployed axis/origin, copied VERBATIM from
// references/cyton_gamma_1500_trac_ik.urdf -- already tilt/origin-corrected.
// MUST be kept in sync with the URDF if it's ever refit.
constexpr Vec3 kJointOrigin[N] = {
    {0.0, 0.0, 0.05315},
    {0.0205, 0.0, 0.12435},
    {-0.02478414, -0.0205, 0.1308452},
    {0.01656849, 0.02722018, 0.11356304},
    {-0.0171, -0.018, 0.09746},
    {0.02765348, 0.01273746, 0.07244612},
    {-0.026255, 0.0, 0.051425},
};
constexpr Vec3 kJointAxisRaw[N] = {
    {0.013792, 0.014877, 0.999794},
    {0.998997, -0.043573, -0.010357},
    {-0.027678, -0.999437, 0.018973},
    {0.999677, -0.024005, 0.008304},
    {0.0, -1.0, 0.0},
    {0.999044, 0.042957, 0.008105},
    {-0.006451, -0.018572, 0.999807},
};

struct FkFrames {
    std::array<Vec3, N> jointPos;   // each joint's own origin, in base_link frame
    std::array<Vec3, N> jointAxis;  // each joint's axis, in base_link frame
    Vec3 chainEndPos;  // position after all 7 joint transforms (pre virtual_endeffector
                        // offset) -- matches final_deployment_fit.py's "pee" exactly, not
                        // the true tool/marker position. Gravity coefficients are only
                        // valid against this same reference point.
};

// Mirrors final_deployment_fit.py's fk_frames(): records each joint's own
// origin/axis BEFORE applying that joint's rotation, matching the Python
// composition order exactly (see homogeneous_transform() in
// calibrate_kinematics.py).
FkFrames computeFkFrames(const std::array<double, N>& jointAnglesRad) {
    FkFrames frames{};
    Mat3 R = identity3();
    Vec3 p{0.0, 0.0, 0.0};
    for (int i = 0; i < N; ++i) {
        const Vec3 axisUnit = normalized(kJointAxisRaw[i]);
        const Vec3 origin = kJointOrigin[i];

        const Vec3 posBeforeThisJoint = add(matVec(R, origin), p);
        const Vec3 axisInBase = matVec(R, axisUnit);
        frames.jointPos[static_cast<std::size_t>(i)] = posBeforeThisJoint;
        frames.jointAxis[static_cast<std::size_t>(i)] = axisInBase;

        const Mat3 jointRotation = axisAngleToMatrix(axisUnit, jointAnglesRad[static_cast<std::size_t>(i)]);
        R = matMul(R, jointRotation);
        p = posBeforeThisJoint;
    }
    frames.chainEndPos = p;
    return frames;
}

// Gravity moment-arm term for joint i: torque about joint i's axis from
// gravity acting at the lumped downstream mass position (see CLAUDE.md,
// "lumped gravity/elastostatic deflection model").
std::array<double, N> gravityMomentArms(const FkFrames& frames) {
    constexpr Vec3 kGravityDir{0.0, 0.0, -1.0};
    std::array<double, N> g{};
    for (int i = 0; i < N; ++i) {
        const Vec3 lever = sub(frames.chainEndPos, frames.jointPos[static_cast<std::size_t>(i)]);
        g[static_cast<std::size_t>(i)] =
            dot(cross(lever, kGravityDir), frames.jointAxis[static_cast<std::size_t>(i)]);
    }
    return g;
}

// ---------------------------------------------------------------------
// Fitted coefficients: calibration/current/final_deployment_fit.py on
// calibration/data/deployed_model_training_dataset_374pose.csv -- the same
// dataset the deployed static corrections were fit on, so these are
// consistent with production (see CLAUDE.md for derivation history).
// ---------------------------------------------------------------------

struct CoupleTerm { int i, j, target; double coeff; };

// Index/pairs match COUPLE_TERMS in final_deployment_fit.py exactly.
constexpr CoupleTerm kCoupleTerms[3] = {
    {SHOULDER_ROLL, SHOULDER_YAW, SHOULDER_YAW, 0.002638},
    {SHOULDER_YAW, ELBOW_YAW, ELBOW_YAW, -0.000372},  // targets elbow_yaw -- excluded at apply time
    {SHOULDER_PITCH, ELBOW_PITCH, ELBOW_PITCH, 0.002293},
};

// shoulder_pitch Fourier term: a*sin(angle) + b*cos(angle), evaluated at
// shoulder_pitch's own angle.
constexpr double kFourierA = -0.002404;
constexpr double kFourierB = 0.012840;

// Lumped gravity-deflection coefficient per joint (rad/m), indexed
// shoulder_roll..wrist_roll. elbow_yaw's is fit but never applied. Smaller
// than earlier-history figures quoted in CLAUDE.md for an older dataset --
// not a contradiction, just less residual left to explain on this cleaner
// dataset (see CLAUDE.md).
constexpr double kGravityCoeff[N] = {
    0.001381, 0.002937, 0.000380, 0.011437, -0.001249, 0.000190, 0.000000,
};

}  // namespace

std::array<double, 7> computeCorrection(const std::array<double, 7>& jointAnglesRad) {
    std::array<double, 7> correction{};

    // Fourier (shoulder_pitch only).
    const double sp = jointAnglesRad[static_cast<std::size_t>(SHOULDER_PITCH)];
    correction[static_cast<std::size_t>(SHOULDER_PITCH)] += kFourierA * std::sin(sp) + kFourierB * std::cos(sp);

    // Coupling -- skip any term targeting elbow_yaw (locked, moot; see
    // header comment).
    for (const CoupleTerm& term : kCoupleTerms) {
        if (term.target == ELBOW_YAW) {
            continue;
        }
        correction[static_cast<std::size_t>(term.target)] +=
            term.coeff * jointAnglesRad[static_cast<std::size_t>(term.i)] *
            jointAnglesRad[static_cast<std::size_t>(term.j)];
    }

    // Gravity -- evaluated with the fourier/coupling corrections already
    // folded in, matching final_deployment_fit.py's predict() ordering.
    // Skip elbow_yaw.
    std::array<double, 7> anglesForGravity = jointAnglesRad;
    for (int i = 0; i < 7; ++i) {
        anglesForGravity[static_cast<std::size_t>(i)] += correction[static_cast<std::size_t>(i)];
    }
    const FkFrames frames = computeFkFrames(anglesForGravity);
    const std::array<double, 7> g = gravityMomentArms(frames);
    for (int i = 0; i < 7; ++i) {
        if (i == ELBOW_YAW) {
            continue;
        }
        correction[static_cast<std::size_t>(i)] += kGravityCoeff[i] * g[static_cast<std::size_t>(i)];
    }

    correction[ELBOW_YAW] = 0.0;
    return correction;
}

}  // namespace pose_dependent_correction
