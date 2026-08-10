import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import calibrate_kinematics as ck

CSV_PATHS = [
    'C:/Users/ConformalUser/Desktop/cyton_setup/build/quick_calibration_test_fixed_elbow_yaw_batch3_newonly.csv',
]
angles_list, pos_list, quat_list = [], [], []
for p in CSV_PATHS:
    a, pos, q = ck.load_poses_from_csv(p)
    angles_list.append(a); pos_list.append(pos); quat_list.append(q)
    print(f'Loaded {len(a)} poses from {p}')
angles_all = np.concatenate(angles_list)
pos_mm_all = np.concatenate(pos_list)
quat_xyzw_all = np.concatenate(quat_list)
n_all = len(angles_all)
print(f'Total combined: {n_all} poses')

SHOULDER_ROLL_IDX, SHOULDER_PITCH_IDX, SHOULDER_YAW_IDX = 0, 1, 2
ELBOW_PITCH_IDX, ELBOW_YAW_IDX, WRIST_PITCH_IDX, WRIST_ROLL_IDX = 3, 4, 5, 6
COUPLE_TERMS = [(SHOULDER_ROLL_IDX, SHOULDER_YAW_IDX, SHOULDER_YAW_IDX), (SHOULDER_YAW_IDX, ELBOW_YAW_IDX, ELBOW_YAW_IDX), (SHOULDER_PITCH_IDX, ELBOW_PITCH_IDX, ELBOW_PITCH_IDX)]
N_COUPLE = len(COUPLE_TERMS); N_GRAVITY = 7
PERP = []
for a in ck.JOINT_AXES:
    a = a / np.linalg.norm(a)
    arbitrary = np.array([1.0,0.0,0.0]) if abs(a[0])<0.9 else np.array([0.0,1.0,0.0])
    u = np.cross(a, arbitrary); u /= np.linalg.norm(u); v = np.cross(a,u)
    PERP.append((u,v))
def tilted_axes(tilt_all):
    axes=[]
    for i in range(ck.N_JOINTS):
        a=ck.JOINT_AXES[i]; u,v=PERP[i]
        p=a+tilt_all[i,0]*u+tilt_all[i,1]*v
        axes.append(p/np.linalg.norm(p))
    return axes
def fk_combined(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes=tilted_axes(tilt_all); T=np.eye(4)
    for i in range(ck.N_JOINTS):
        origin=ck.JOINT_ORIGINS_M[i].copy()
        if i==WRIST_PITCH_IDX: origin=origin+o_wp
        if i==ELBOW_PITCH_IDX: origin=origin+o_elbow
        if i==SHOULDER_YAW_IDX: origin=origin+np.array([o_sy_xz[0],0.0,o_sy_xz[1]])
        R=Rotation.from_rotvec(axes[i]*ja[i]).as_matrix()
        T=T@ck.homogeneous_transform(R,origin)
    return T
def fk_frames(ja, tilt_all, o_elbow, o_sy_xz, o_wp):
    axes=tilted_axes(tilt_all); T=np.eye(4); pl=[]; al=[]
    for i in range(ck.N_JOINTS):
        origin=ck.JOINT_ORIGINS_M[i].copy()
        if i==WRIST_PITCH_IDX: origin=origin+o_wp
        if i==ELBOW_PITCH_IDX: origin=origin+o_elbow
        if i==SHOULDER_YAW_IDX: origin=origin+np.array([o_sy_xz[0],0.0,o_sy_xz[1]])
        p=T[:3,:3]@origin+T[:3,3]; a=T[:3,:3]@axes[i]
        pl.append(p); al.append(a)
        R=Rotation.from_rotvec(axes[i]*ja[i]).as_matrix()
        T=T@ck.homogeneous_transform(R,origin)
    return T,pl,al
GDIR=np.array([0.0,0.0,-1.0])
def gterms(ca, tilt_all, o_elbow, o_sy_xz, o_wp):
    T,pl,al=fk_frames(ca,tilt_all,o_elbow,o_sy_xz,o_wp); pee=T[:3,3]
    g=np.zeros(7)
    for i in range(7):
        lever=pee-pl[i]; g[i]=np.dot(np.cross(lever,GDIR),al[i])
    return g
N_TILT,N_SCALE=14,7
OFF_SCALE=19; OFF_TILT=OFF_SCALE+N_SCALE; OFF_OE=OFF_TILT+N_TILT; OFF_OSY=OFF_OE+3; OFF_OWP=OFF_OSY+2; OFF_F=OFF_OWP+3; OFF_C=OFF_F+2; OFF_G=OFF_C+N_COUPLE
TOTAL=OFF_G+N_GRAVITY
def unpack(x):
    bp=x[0:19]; js=x[OFF_SCALE:OFF_SCALE+N_SCALE]; tl=x[OFF_TILT:OFF_TILT+N_TILT].reshape(7,2)
    oe=x[OFF_OE:OFF_OE+3]; osy=x[OFF_OSY:OFF_OSY+2]; owp=x[OFF_OWP:OFF_OWP+3]
    fab=x[OFF_F:OFF_F+2]; cc=x[OFF_C:OFF_C+N_COUPLE]; gc=x[OFF_G:OFF_G+N_GRAVITY]
    return bp,js,tl,oe,osy,owp,fab,cc,gc
def predict(ar,bp,js,tl,oe,osy,owp,fab,cc,gc):
    params=ck.unpack_params(bp)
    ca=(ar*js+params.joint_offsets).copy()
    rsp=ar[SHOULDER_PITCH_IDX]
    ca[SHOULDER_PITCH_IDX]+=fab[0]*np.sin(rsp)+fab[1]*np.cos(rsp)
    for k,(i,j,t) in enumerate(COUPLE_TERMS):
        ca[t]+=cc[k]*ar[i]*ar[j]
    g=gterms(ca,tl,oe,osy,owp)
    cag=ca+gc*g
    Tfk=fk_combined(cag,tl,oe,osy,owp)
    Ttool=ck.build_tool_transform(params.tool_xyz,params.tool_rpy)
    Tbase=ck.build_base_transform(params.base_xyz,params.base_rpy)
    return Tbase@Tfk@Ttool
TW,TS=5.0,np.radians(8.0); SW=20.0; FW,FS=5.0,np.radians(5.0); CW,CS,CB=5.0,0.1,0.3; GW,GS,GB=5.0,0.1,0.5
def make_res(angles,pos,quat):
    n=len(angles)
    def res(x):
        bp,js,tl,oe,osy,owp,fab,cc,gc=unpack(x); params=ck.unpack_params(bp)
        pr=np.zeros((n,3)); orr=np.zeros((n,3))
        for i in range(n):
            Tp=predict(angles[i],bp,js,tl,oe,osy,owp,fab,cc,gc)
            pr[i]=pos[i]-Tp[:3,3]*1000.0
            Rp=Tp[:3,:3]; Rm=Rotation.from_quat(quat[i]).as_matrix()
            orr[i]=Rotation.from_matrix(Rp.T@Rm).as_rotvec()*100.0*0.3
        reg=np.concatenate([params.joint_offsets*(8.0/np.radians(8.0)),params.tool_xyz*(1.0/0.01),params.tool_rpy*(1.0/np.radians(10.0)),tl.ravel()*(TW/TS),(js-1.0)*SW,fab*(FW/FS),cc*(CW/CS),gc*(GW/GS)])
        return np.concatenate([pr.ravel(),orr.ravel(),reg])
    return res
def bounds():
    lo=np.concatenate([-np.radians(15.0)*np.ones(7),-0.10*np.ones(3),[-np.inf]*3,[-np.inf]*3,[-np.inf]*3,0.90*np.ones(N_SCALE),-np.radians(15.0)*np.ones(N_TILT),-0.05*np.ones(3),-0.05*np.ones(2),-0.05*np.ones(3),-np.radians(15.0)*np.ones(2),-CB*np.ones(N_COUPLE),-GB*np.ones(N_GRAVITY)])
    up=np.concatenate([np.radians(15.0)*np.ones(7),0.10*np.ones(3),[np.inf]*3,[np.inf]*3,[np.inf]*3,1.10*np.ones(N_SCALE),np.radians(15.0)*np.ones(N_TILT),0.05*np.ones(3),0.05*np.ones(2),0.05*np.ones(3),np.radians(15.0)*np.ones(2),CB*np.ones(N_COUPLE),GB*np.ones(N_GRAVITY)])
    return lo,up
def x0(angles,pos,quat):
    bx,br=ck.initial_base_guess(angles[0],pos[0],quat[0])
    return np.concatenate([ck.pack_params(np.zeros(7),np.zeros(3),np.zeros(3),bx,br),np.ones(N_SCALE),np.zeros(N_TILT+3+2+3+2+N_COUPLE+N_GRAVITY)])
def fit(angles,pos,quat):
    lo,up=bounds()
    return least_squares(make_res(angles,pos,quat), x0(angles,pos,quat), method='trf', x_scale='jac', bounds=(lo,up), max_nfev=8000)

print(f'Fitting {TOTAL}-param model on {n_all} combined poses...')
result = fit(angles_all, pos_mm_all, quat_xyzw_all)
bp,js,tl,oe,osy,owp,fab,cc,gc = unpack(result.x)
params = ck.unpack_params(bp)

e = np.zeros(n_all)
for i in range(n_all):
    Tp = predict(angles_all[i],bp,js,tl,oe,osy,owp,fab,cc,gc)
    e[i] = np.linalg.norm(pos_mm_all[i]-Tp[:3,3]*1000.0)
print(f'Full-dataset in-sample RMS: {np.sqrt(np.mean(e**2)):.2f}mm')
print(f'Full-dataset mean error: {np.mean(e):.2f}mm, max: {np.max(e):.2f}mm')
_,s,_ = np.linalg.svd(result.jac)
print(f'Condition number: {s[0]/s[-1]:.1f}')

# Blocked (contiguous pose_id) 8-fold CV -- the project-established honest
# validation methodology (random splits leak given trajectory-structured data).
N_FOLDS = 8
fold_bounds = np.linspace(0, n_all, N_FOLDS + 1).astype(int)
fold_test_sq_errs = []
print('\n=== BLOCKED (contiguous) 8-FOLD CV ===')
for f in range(N_FOLDS):
    lo_i, hi_i = fold_bounds[f], fold_bounds[f + 1]
    test_mask = np.zeros(n_all, dtype=bool)
    test_mask[lo_i:hi_i] = True
    train_mask = ~test_mask
    if test_mask.sum() < 3 or train_mask.sum() < 20:
        continue
    r = fit(angles_all[train_mask], pos_mm_all[train_mask], quat_xyzw_all[train_mask])
    bp_f, js_f, tl_f, oe_f, osy_f, owp_f, fab_f, cc_f, gc_f = unpack(r.x)
    train_e = np.zeros(train_mask.sum())
    for i, idx in enumerate(np.where(train_mask)[0]):
        Tp = predict(angles_all[idx], bp_f, js_f, tl_f, oe_f, osy_f, owp_f, fab_f, cc_f, gc_f)
        train_e[i] = np.linalg.norm(pos_mm_all[idx] - Tp[:3, 3] * 1000.0)
    test_e = np.zeros(test_mask.sum())
    for i, idx in enumerate(np.where(test_mask)[0]):
        Tp = predict(angles_all[idx], bp_f, js_f, tl_f, oe_f, osy_f, owp_f, fab_f, cc_f, gc_f)
        test_e[i] = np.linalg.norm(pos_mm_all[idx] - Tp[:3, 3] * 1000.0)
    fold_test_sq_errs.append(test_e ** 2)
    print(f'  fold {f} (idx {lo_i}-{hi_i}): train={np.sqrt(np.mean(train_e**2)):.2f}mm '
          f'test={np.sqrt(np.mean(test_e**2)):.2f}mm  n_test={test_mask.sum()}')
pooled = np.concatenate(fold_test_sq_errs)
print(f'\nPooled blocked-CV test RMS: {np.sqrt(np.mean(pooled)):.2f}mm')

JOINT_NAMES = ['shoulder_roll','shoulder_pitch','shoulder_yaw','elbow_pitch','elbow_yaw','wrist_pitch','wrist_roll']
print('\n=== JOINT OFFSETS (deg) ===')
for i,name in enumerate(JOINT_NAMES):
    print(f'  {name}: {np.degrees(params.joint_offsets[i]):+.4f} deg')
print('\n=== JOINT SCALES ===')
for i,name in enumerate(JOINT_NAMES):
    print(f'  {name}: {js[i]:.6f}')
print('\n=== ORIGIN CORRECTIONS (mm) ===')
print(f'  elbow_pitch (idx3) xyz: {oe*1000}')
print(f'  shoulder_yaw (idx2) x,z: {osy*1000}')
print(f'  wrist_pitch (idx5) xyz: {owp*1000}')
print('\n=== TOOL FRAME (virtual_endeffector -> moving marker) ===')
print(f'  xyz mm: {params.tool_xyz*1000}')
print(f'  rpy deg: {np.degrees(params.tool_rpy)}')
print('\n=== GRAVITY (rad/m) ===')
for i,name in enumerate(JOINT_NAMES):
    print(f'  {name}: {gc[i]:.6f}')
