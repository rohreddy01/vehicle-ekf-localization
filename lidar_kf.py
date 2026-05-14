"""
LiDAR-Based Odometry with Kalman Filter Integration
Author: Rohit Reddy Udumula

This project extends the EKF vehicle localization system (Project 1) by replacing
GNSS/wheel-odometry with simulated LiDAR odometry (NDT/ICP-style scan matching).
A linear Kalman Filter (KF) is used to fuse LiDAR odometry pose estimates with
IMU predictions, demonstrating noise reduction and drift correction.

LiDAR Odometry Model:
  - Simulates NDT (Normal Distributions Transform) scan-matching output:
    incremental pose (dx, dy, dpsi) with:
      * Gaussian white noise (sensor quantisation / matching uncertainty)
      * Multiplicative drift (systematic error proportional to distance/rotation)
      * Occasional outlier spikes (simulating featureless / dynamic scenes)
  - Output: accumulated pose estimate (px, py, psi) at 10 Hz
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os, json, time

# ── Output directories ────────────────────────────────────────────────────────
base_dir   = os.path.dirname(os.path.abspath(__file__))
plot_dir   = os.path.join(base_dir, "plots_p2")
output_dir = os.path.join(base_dir, "output_p2")
os.makedirs(plot_dir,   exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

# ── Simulation parameters ─────────────────────────────────────────────────────
DT       = 0.1       # filter / LiDAR step (10 Hz)
DURATION = 120.0
N        = int(DURATION / DT)   # 1200 steps

# ── IMU noise (same as Project 1) ─────────────────────────────────────────────
IMU_ACCEL_NOISE = 0.3    # m/s²
IMU_GYRO_NOISE  = 0.02   # rad/s

# ── LiDAR odometry noise parameters ──────────────────────────────────────────
# These mimic realistic NDT output characteristics:
LIDAR_XY_NOISE      = 0.08   # m  – Gaussian matching uncertainty per step
LIDAR_PSI_NOISE     = 0.005  # rad – heading matching uncertainty per step
LIDAR_DRIFT_XY      = 0.002  # fractional drift per meter travelled (0.2 % / m)
LIDAR_DRIFT_PSI     = 0.001  # fractional drift per radian turned
LIDAR_OUTLIER_PROB  = 0.02   # 2 % chance of a gross outlier per step
LIDAR_OUTLIER_SCALE = 5.0    # outlier magnitude multiplier

# ── Sensitivity sweep values ──────────────────────────────────────────────────
SWEEP_LIDAR_XY_NOISE  = [0.02, 0.08, 0.25]
SWEEP_LIDAR_PSI_NOISE = [0.002, 0.005, 0.02]
SWEEP_LIDAR_DRIFT     = [0.0005, 0.002, 0.008]
SWEEP_DT              = [0.05, 0.1, 0.5]

# ── LiDAR update rate sweep (every N steps, IMU runs every step) ──────────
SWEEP_LIDAR_EVERY     = [1, 5, 10, 20]   # 10 Hz, 2 Hz, 1 Hz, 0.5 Hz


# ═════════════════════════════════════════════════════════════════════════════
#  1. Ground-Truth Trajectory  (identical to Project 1 for comparability)
# ═════════════════════════════════════════════════════════════════════════════
def generate_trajectory():
    t   = np.arange(N) * DT
    px  = np.zeros(N); py  = np.zeros(N)
    psi = np.zeros(N); vx  = np.zeros(N)
    ax  = np.zeros(N); yr  = np.zeros(N)

    for i in range(N):
        ti = t[i]
        if   ti < 10:  vx[i], yr[i] = 5.0*(ti/10),          0.0
        elif ti < 20:  vx[i], yr[i] = 5.0,                   0.0
        elif ti < 35:  vx[i], yr[i] = 5.0,                   0.15
        elif ti < 50:  vx[i], yr[i] = 5.0,                  -0.15
        elif ti < 60:  vx[i], yr[i] = 5.0,                   0.0
        elif ti < 75:  vx[i], yr[i] = 6.0,                   0.10
        elif ti < 90:  vx[i], yr[i] = 4.0,                  -0.12
        else:          vx[i], yr[i] = 5.0*(1-(ti-90)/30),    0.0

        if i > 0:
            psi[i] = psi[i-1] + yr[i-1] * DT
            px[i]  = px[i-1]  + vx[i-1] * np.cos(psi[i-1]) * DT
            py[i]  = py[i-1]  + vx[i-1] * np.sin(psi[i-1]) * DT
        ax[i] = (vx[i] - vx[i-1]) / DT if i > 0 else 0.0

    return t, px, py, psi, vx, np.zeros(N), ax, yr


# ═════════════════════════════════════════════════════════════════════════════
#  2. Sensor Simulation
# ═════════════════════════════════════════════════════════════════════════════
def simulate_imu(ax, yr):
    """High-rate IMU sampled at DT."""
    return (ax + np.random.normal(0, IMU_ACCEL_NOISE, N),
            yr + np.random.normal(0, IMU_GYRO_NOISE,  N))


def simulate_lidar_odometry(px_gt, py_gt, psi_gt, vx_gt, yr_gt,
                             xy_noise=LIDAR_XY_NOISE,
                             psi_noise=LIDAR_PSI_NOISE,
                             drift_xy=LIDAR_DRIFT_XY,
                             drift_psi=LIDAR_DRIFT_PSI,
                             outlier_prob=LIDAR_OUTLIER_PROB,
                             outlier_scale=LIDAR_OUTLIER_SCALE,
                             n_override=None):
    """
    Simulate NDT/ICP-style LiDAR odometry output.

    Model:
      dpose_true  = true incremental motion in body frame
      dpose_noisy = dpose_true
                    + Gaussian white noise  (matching uncertainty)
                    + multiplicative drift  (proportional to motion magnitude)
                    + sporadic outlier spike (featureless or dynamic scene)

    The accumulated pose from noisy increments is the 'raw LiDAR odometry' signal
    that the Kalman Filter will then correct.
    """
    _n = n_override if n_override is not None else len(px_gt)
    lio_px  = np.zeros(_n)
    lio_py  = np.zeros(_n)
    lio_psi = np.zeros(_n)

    # Start from ground truth
    lio_px[0]  = px_gt[0]
    lio_py[0]  = py_gt[0]
    lio_psi[0] = psi_gt[0]

    for i in range(1, _n):
        # True increments (navigation frame)
        dpx_true  = px_gt[i]  - px_gt[i-1]
        dpy_true  = py_gt[i]  - py_gt[i-1]
        dpsi_true = psi_gt[i] - psi_gt[i-1]

        dist = np.sqrt(dpx_true**2 + dpy_true**2)

        # Gaussian white noise
        nx  = np.random.normal(0, xy_noise)
        ny  = np.random.normal(0, xy_noise)
        npsi= np.random.normal(0, psi_noise)

        # Multiplicative drift (systematic bias proportional to motion)
        drift_x   = drift_xy  * dist    * np.random.choice([-1, 1])
        drift_y   = drift_xy  * dist    * np.random.choice([-1, 1])
        drift_p   = drift_psi * abs(dpsi_true) * np.random.choice([-1, 1])

        # Sporadic outlier (failed scan match)
        if np.random.rand() < outlier_prob:
            nx   *= outlier_scale
            ny   *= outlier_scale
            npsi *= outlier_scale

        dpx_noisy  = dpx_true  + nx   + drift_x
        dpy_noisy  = dpy_true  + ny   + drift_y
        dpsi_noisy = dpsi_true + npsi + drift_p

        lio_px[i]  = lio_px[i-1]  + dpx_noisy
        lio_py[i]  = lio_py[i-1]  + dpy_noisy
        lio_psi[i] = lio_psi[i-1] + dpsi_noisy

    return lio_px, lio_py, lio_psi


# ═════════════════════════════════════════════════════════════════════════════
#  3. Linear Kalman Filter (fuses IMU prediction + LiDAR pose update)
# ═════════════════════════════════════════════════════════════════════════════
class LiDARKalmanFilter:
    """
    State: x = [px, py, psi, vx, vy]   (5-D)

    Prediction: linear kinematic model driven by IMU
      px'  = px + (vx*cos(psi) - vy*sin(psi))*dt  ← linearised at current psi
      py'  = py + (vx*sin(psi) + vy*cos(psi))*dt
      psi' = psi + yr_imu * dt
      vx'  = vx + ax_imu * dt
      vy'  = vy

    Measurement: LiDAR odometry provides [px, py, psi]
      H = [[1,0,0,0,0],
           [0,1,0,0,0],
           [0,0,1,0,0]]

    This is a *linear* KF (not EKF) because we pre-linearise the state
    transition at each step (same Jacobian as in EKF, but reused directly
    as F without re-linearisation during update).  This is valid because
    the LiDAR observation is already expressed in navigation-frame coordinates.
    """

    def __init__(self, lidar_xy_noise=LIDAR_XY_NOISE, lidar_psi_noise=LIDAR_PSI_NOISE):
        self.x = np.zeros(5)
        # Prior covariance — large positional uncertainty at start
        self.P = np.diag([5**2, 5**2, (0.5)**2, 3**2, 1**2])

        # Process noise Q — model uncertainty from IMU integration
        self.Q = np.diag([
            0.05**2,   # px
            0.05**2,   # py
            0.005**2,  # psi
            0.5**2,    # vx
            0.1**2,    # vy
        ])

        # Measurement noise R — LiDAR odometry uncertainty
        # Accumulated over DT, so per-step variance = per-step noise^2
        self.R_lidar = np.diag([
            lidar_xy_noise**2,   # px measurement noise
            lidar_xy_noise**2,   # py measurement noise
            lidar_psi_noise**2,  # psi measurement noise
        ])

        # LiDAR observation matrix H
        self.H = np.zeros((3, 5))
        self.H[0, 0] = 1.0   # observe px
        self.H[1, 1] = 1.0   # observe py
        self.H[2, 2] = 1.0   # observe psi

    def predict(self, ax_imu, yr_imu, dt=DT):
        """IMU-driven linear state prediction with Jacobian linearisation."""
        px, py, psi, vx, vy = self.x

        # State transition (linearised about current psi)
        self.x = np.array([
            px  + (vx*np.cos(psi) - vy*np.sin(psi)) * dt,
            py  + (vx*np.sin(psi) + vy*np.cos(psi)) * dt,
            psi + yr_imu * dt,
            vx  + ax_imu * dt,
            vy,
        ])

        # State transition Jacobian F = df/dx
        F = np.eye(5)
        F[0, 2] = (-vx*np.sin(psi) - vy*np.cos(psi)) * dt
        F[0, 3] =  np.cos(psi) * dt
        F[0, 4] = -np.sin(psi) * dt
        F[1, 2] = ( vx*np.cos(psi) - vy*np.sin(psi)) * dt
        F[1, 3] =  np.sin(psi) * dt
        F[1, 4] =  np.cos(psi) * dt

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update_lidar(self, z_lidar):
        """
        Standard KF update with LiDAR pose measurement z = [px, py, psi].
        Uses Joseph-form covariance update for numerical stability.
        """
        H = self.H
        y = z_lidar - H @ self.x          # innovation
        S = H @ self.P @ H.T + self.R_lidar  # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        I_KH   = np.eye(5) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_lidar @ K.T
        return self.x.copy(), K.copy()


# ═════════════════════════════════════════════════════════════════════════════
#  4. Dead Reckoning baseline (IMU only)
# ═════════════════════════════════════════════════════════════════════════════
class DeadReckoning:
    def __init__(self):
        self.x = np.zeros(5)

    def step(self, ax_imu, yr_imu, dt=DT):
        px, py, psi, vx, vy = self.x
        self.x = np.array([
            px  + vx*np.cos(psi)*dt,
            py  + vx*np.sin(psi)*dt,
            psi + yr_imu*dt,
            vx  + ax_imu*dt,
            vy,
        ])
        return self.x.copy()


# ═════════════════════════════════════════════════════════════════════════════
#  5. Main simulation loop
# ═════════════════════════════════════════════════════════════════════════════
def run_simulation(lidar_xy_noise=LIDAR_XY_NOISE,
                   lidar_every=1,
                   lidar_psi_noise=LIDAR_PSI_NOISE,
                   drift_xy=LIDAR_DRIFT_XY,
                   drift_psi=LIDAR_DRIFT_PSI,
                   dt=DT,
                   seed=42):
    """
    Run the full pipeline and return result arrays + timing stats.
    Accepts parameter overrides for sensitivity sweeps.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)

    n_steps = int(DURATION / dt)

    # Re-generate trajectory at requested dt
    t_arr = np.arange(n_steps) * dt
    px_gt  = np.zeros(n_steps); py_gt  = np.zeros(n_steps)
    psi_gt = np.zeros(n_steps); vx_gt  = np.zeros(n_steps)
    ax_gt  = np.zeros(n_steps); yr_gt  = np.zeros(n_steps)

    for i in range(n_steps):
        ti = t_arr[i]
        if   ti < 10:  vx_gt[i], yr_gt[i] = 5.0*(ti/10),        0.0
        elif ti < 20:  vx_gt[i], yr_gt[i] = 5.0,                 0.0
        elif ti < 35:  vx_gt[i], yr_gt[i] = 5.0,                 0.15
        elif ti < 50:  vx_gt[i], yr_gt[i] = 5.0,                -0.15
        elif ti < 60:  vx_gt[i], yr_gt[i] = 5.0,                 0.0
        elif ti < 75:  vx_gt[i], yr_gt[i] = 6.0,                 0.10
        elif ti < 90:  vx_gt[i], yr_gt[i] = 4.0,                -0.12
        else:          vx_gt[i], yr_gt[i] = 5.0*(1-(ti-90)/30),  0.0
        if i > 0:
            psi_gt[i] = psi_gt[i-1] + yr_gt[i-1]*dt
            px_gt[i]  = px_gt[i-1]  + vx_gt[i-1]*np.cos(psi_gt[i-1])*dt
            py_gt[i]  = py_gt[i-1]  + vx_gt[i-1]*np.sin(psi_gt[i-1])*dt
        ax_gt[i] = (vx_gt[i]-vx_gt[i-1])/dt if i > 0 else 0.0

    # Sensor data
    imu_accel = ax_gt + np.random.normal(0, IMU_ACCEL_NOISE, n_steps)
    imu_gyro  = yr_gt + np.random.normal(0, IMU_GYRO_NOISE,  n_steps)

    lio_px, lio_py, lio_psi = simulate_lidar_odometry(
        px_gt, py_gt, psi_gt, vx_gt, yr_gt,
        xy_noise=lidar_xy_noise, psi_noise=lidar_psi_noise,
        drift_xy=drift_xy, drift_psi=drift_psi, n_override=n_steps)

    # Initialise estimators
    kf = LiDARKalmanFilter(lidar_xy_noise=lidar_xy_noise,
                           lidar_psi_noise=lidar_psi_noise)
    kf.x = np.array([px_gt[0], py_gt[0], psi_gt[0], vx_gt[0], 0.0])

    dr = DeadReckoning()
    dr.x = kf.x.copy()

    kf_states  = np.zeros((n_steps, 5))
    dr_states  = np.zeros((n_steps, 5))
    kalman_gains = []

    t0 = time.perf_counter()
    for i in range(n_steps):
        kf.predict(imu_accel[i], imu_gyro[i], dt=dt)
        dr.step(imu_accel[i], imu_gyro[i], dt=dt)

        # LiDAR update only when a new scan is available (rate control)
        if i % lidar_every == 0:
            idx = min(i, len(lio_px)-1)
            z_lidar = np.array([lio_px[idx], lio_py[idx], lio_psi[idx]])
            _, K = kf.update_lidar(z_lidar)
            kalman_gains.append(K[0, 0])   # px channel gain
        else:
            kalman_gains.append(float('nan'))  # no update this step

        kf_states[i] = kf.x
        dr_states[i] = dr.x
    t1 = time.perf_counter()

    compute_time_ms = (t1 - t0) * 1000.0

    np.random.set_state(rng_state)

    return dict(
        t=t_arr, n=n_steps, dt=dt,
        px_gt=px_gt, py_gt=py_gt, psi_gt=psi_gt, vx_gt=vx_gt,
        lio_px=lio_px, lio_py=lio_py, lio_psi=lio_psi,
        kf_states=kf_states, dr_states=dr_states,
        kalman_gains=np.array(kalman_gains),
        compute_time_ms=compute_time_ms,
        lidar_every=lidar_every,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  6. Error metrics helper
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(res):
    n = res['n']
    px_gt, py_gt, psi_gt = res['px_gt'][:n], res['py_gt'][:n], res['psi_gt'][:n]
    kf  = res['kf_states'][:n]
    dr  = res['dr_states'][:n]
    lio = np.stack([res['lio_px'][:n], res['lio_py'][:n]], axis=1)

    kf_pos  = np.sqrt((kf[:,0]-px_gt)**2 + (kf[:,1]-py_gt)**2)
    dr_pos  = np.sqrt((dr[:,0]-px_gt)**2 + (dr[:,1]-py_gt)**2)
    lio_pos = np.sqrt((lio[:,0]-px_gt)**2 + (lio[:,1]-py_gt)**2)
    kf_head = np.abs(np.unwrap(kf[:,2]) - np.unwrap(psi_gt)) * 180/np.pi
    dr_head = np.abs(np.unwrap(dr[:,2]) - np.unwrap(psi_gt)) * 180/np.pi

    def rmse(e): return float(np.sqrt(np.mean(e**2)))
    return dict(
        kf_rmse=rmse(kf_pos),   kf_mean=float(kf_pos.mean()),   kf_max=float(kf_pos.max()),
        dr_rmse=rmse(dr_pos),   dr_mean=float(dr_pos.mean()),   dr_max=float(dr_pos.max()),
        lio_rmse=rmse(lio_pos), lio_mean=float(lio_pos.mean()), lio_max=float(lio_pos.max()),
        kf_head_rmse=rmse(kf_head), dr_head_rmse=rmse(dr_head),
        kf_pos_err=kf_pos, dr_pos_err=dr_pos, lio_pos_err=lio_pos,
        kf_head_err=kf_head, dr_head_err=dr_head,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  7. Plotting
# ═════════════════════════════════════════════════════════════════════════════
def plot_trajectory(res, m, tag=''):
    fig, ax = plt.subplots(figsize=(12, 7))
    t = res['t']
    ax.plot(res['px_gt'],      res['py_gt'],      'k-',  lw=2.5, label='Ground Truth',            zorder=5)
    ax.plot(res['kf_states'][:,0], res['kf_states'][:,1], 'b-',  lw=1.8, alpha=0.9, label='KF (fused)',  zorder=4)
    ax.plot(res['dr_states'][:,0], res['dr_states'][:,1], 'r--', lw=1.2, alpha=0.7, label='Dead Reckoning (IMU)',  zorder=3)
    ax.plot(res['lio_px'],     res['lio_py'],     'g-',  lw=1.0, alpha=0.6, label='LiDAR Odometry (raw)',zorder=2)
    ax.scatter([res['px_gt'][0]], [res['py_gt'][0]], c='black', s=100, marker='*', zorder=6, label='Start')
    ax.set_xlabel('East Position [m]', fontsize=12)
    ax.set_ylabel('North Position [m]', fontsize=12)
    ax.set_title(f'LiDAR-KF Vehicle Localization — Trajectory{tag}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    plt.tight_layout()
    fname = os.path.join(plot_dir, f'fig1_trajectory{tag}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_errors(res, m, tag=''):
    t = res['t']
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(t, m['kf_pos_err'],  'b-',  lw=1.5, label='KF Fusion')
    axes[0].plot(t, m['dr_pos_err'],  'r--', lw=1.2, alpha=0.8, label='Dead Reckoning')
    axes[0].plot(t, m['lio_pos_err'], 'g-',  lw=1.0, alpha=0.6, label='Raw LiDAR Odometry')
    axes[0].set_ylabel('Position Error [m]', fontsize=11)
    axes[0].set_title('Position Error vs Time', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, m['kf_head_err'],  'b-',  lw=1.5, label='KF Fusion')
    axes[1].plot(t, m['dr_head_err'],  'r--', lw=1.2, alpha=0.8, label='Dead Reckoning')
    axes[1].set_xlabel('Time [s]', fontsize=11)
    axes[1].set_ylabel('Heading Error [deg]', fontsize=11)
    axes[1].set_title('Heading Error vs Time', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(plot_dir, f'fig2_errors{tag}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_rmse_bar(m, tag=''):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    methods   = ['Dead\nReckoning', 'Raw LiDAR\nOdometry', 'KF\nFusion']
    pos_rmse  = [m['dr_rmse'], m['lio_rmse'], m['kf_rmse']]
    colors    = ['#e74c3c', '#f39c12', '#3498db']

    bars = axes[0].bar(methods, pos_rmse, color=colors, width=0.5, edgecolor='black')
    axes[0].set_ylabel('RMSE [m]', fontsize=12)
    axes[0].set_title('Position RMSE Comparison', fontsize=12, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, pos_rmse):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                     f'{val:.2f}m', ha='center', va='bottom', fontweight='bold', fontsize=10)

    head_vals = [m['dr_head_rmse'], 0, m['kf_head_rmse']]
    methods2  = ['Dead\nReckoning', 'LiDAR\n(N/A)', 'KF\nFusion']
    colors2   = ['#e74c3c', '#cccccc', '#3498db']
    bars2 = axes[1].bar(methods2, head_vals, color=colors2, width=0.5, edgecolor='black')
    axes[1].set_ylabel('RMSE [degrees]', fontsize=12)
    axes[1].set_title('Heading RMSE Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)
    for bar, val, lbl in zip(bars2, head_vals, [m['dr_head_rmse'], None, m['kf_head_rmse']]):
        if lbl is not None:
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                         f'{lbl:.2f}°', ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.tight_layout()
    fname = os.path.join(plot_dir, f'fig3_rmse_bar{tag}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_lidar_noise_characterization(res):
    """Analyse and visualise LiDAR odometry noise distribution."""
    px_gt, py_gt, psi_gt = res['px_gt'], res['py_gt'], res['psi_gt']
    lio_px, lio_py, lio_psi = res['lio_px'], res['lio_py'], res['lio_psi']

    err_x   = lio_px  - px_gt
    err_y   = lio_py  - py_gt
    err_psi = (lio_psi - psi_gt) * 180/np.pi

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Time-series errors
    t = res['t']
    axes[0,0].plot(t, err_x,   'r-', lw=0.8, alpha=0.8)
    axes[0,0].set_title('LiDAR X Error vs Time', fontsize=11, fontweight='bold')
    axes[0,0].set_ylabel('Error [m]'); axes[0,0].set_xlabel('Time [s]')
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t, err_y,   'b-', lw=0.8, alpha=0.8)
    axes[0,1].set_title('LiDAR Y Error vs Time', fontsize=11, fontweight='bold')
    axes[0,1].set_ylabel('Error [m]'); axes[0,1].set_xlabel('Time [s]')
    axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(t, err_psi, 'g-', lw=0.8, alpha=0.8)
    axes[0,2].set_title('LiDAR Heading Error vs Time', fontsize=11, fontweight='bold')
    axes[0,2].set_ylabel('Error [deg]'); axes[0,2].set_xlabel('Time [s]')
    axes[0,2].grid(True, alpha=0.3)

    # Histograms with Gaussian fit
    from scipy import stats as scipy_stats

    def plot_hist_fit(ax, data, xlabel, color):
        ax.hist(data, bins=40, density=True, color=color, alpha=0.6, edgecolor='white')
        mu, sigma = data.mean(), data.std()
        xs = np.linspace(data.min(), data.max(), 200)
        ax.plot(xs, scipy_stats.norm.pdf(xs, mu, sigma), 'k-', lw=2, label=f'N({mu:.3f},{sigma:.3f})')
        ax.set_xlabel(xlabel); ax.set_ylabel('Density')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plot_hist_fit(axes[1,0], err_x,   'X Error [m]',    'red')
    axes[1,0].set_title('LiDAR X Error Distribution', fontsize=11, fontweight='bold')

    plot_hist_fit(axes[1,1], err_y,   'Y Error [m]',    'blue')
    axes[1,1].set_title('LiDAR Y Error Distribution', fontsize=11, fontweight='bold')

    plot_hist_fit(axes[1,2], err_psi, 'Heading Error [deg]', 'green')
    axes[1,2].set_title('LiDAR Heading Error Distribution', fontsize=11, fontweight='bold')

    plt.suptitle('LiDAR Odometry Noise Characterization', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'fig4_lidar_noise.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_kalman_gain(res):
    """Show Kalman gain evolution (how much the filter trusts LiDAR vs. model)."""
    t = res['t']
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, res['kalman_gains'], 'b-', lw=1.2, alpha=0.85)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Kalman Gain K[px, px]', fontsize=11)
    ax.set_title('Kalman Gain Evolution (px channel)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'fig5_kalman_gain.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_sensitivity_sweep(param_name, param_values, rmse_kf, rmse_lio, label_fmt, xlabel):
    """Generic sensitivity sweep plot."""
    x = np.arange(len(param_values))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, rmse_kf,  w, label='KF Fusion',          color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + w/2, rmse_lio, w, label='Raw LiDAR Odometry', color='#f39c12', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([label_fmt.format(v) for v in param_values], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Position RMSE [m]', fontsize=11)
    ax.set_title(f'Sensitivity: {xlabel}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, axis='y', alpha=0.3)
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    fname = os.path.join(plot_dir, f'fig_sweep_{param_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_accuracy_vs_compute(dt_vals, rmse_kf_vals, compute_vals):
    """Trade-off: accuracy vs computational cost (time step variation)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    c1 = '#3498db'
    c2 = '#e74c3c'
    ax1.plot(dt_vals, rmse_kf_vals, 'o-', color=c1, lw=2, markersize=8, label='KF RMSE [m]')
    ax1.set_xlabel('Integration Time Step Δt [s]', fontsize=11)
    ax1.set_ylabel('Position RMSE [m]', fontsize=11, color=c1)
    ax1.tick_params(axis='y', labelcolor=c1)
    ax2 = ax1.twinx()
    ax2.plot(dt_vals, compute_vals, 's--', color=c2, lw=2, markersize=8, label='Compute [ms]')
    ax2.set_ylabel('Compute Time [ms / 120 s run]', fontsize=11, color=c2)
    ax2.tick_params(axis='y', labelcolor=c2)
    ax1.set_title('Accuracy vs Computational Cost Trade-off', fontsize=12, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'fig_accuracy_vs_compute.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ═════════════════════════════════════════════════════════════════════════════
#  8. Run everything
# ═════════════════════════════════════════════════════════════════════════════

def plot_intentional_noise_trajectories(results_by_noise, noise_values):
    """Side-by-side trajectories as LiDAR XY noise increases."""
    n = len(noise_values)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), sharey=True)
    colors = {'gt': 'black', 'kf': '#3498db', 'lio': '#e67e22'}
    for ax, res, noise in zip(axes, results_by_noise, noise_values):
        mm = compute_metrics(res)
        ax.plot(res['px_gt'], res['py_gt'], '-', color=colors['gt'], lw=2.0, label='Ground Truth', zorder=5)
        ax.plot(res['kf_states'][:,0], res['kf_states'][:,1], '-', color=colors['kf'], lw=1.5,
                alpha=0.9, label=f"KF ({mm['kf_rmse']:.2f}m)", zorder=4)
        ax.plot(res['lio_px'], res['lio_py'], '-', color=colors['lio'], lw=1.0,
                alpha=0.7, label=f"LiO ({mm['lio_rmse']:.2f}m)", zorder=3)
        ax.set_title(f'sigma_xy = {noise:.2f} m', fontsize=11, fontweight='bold')
        ax.set_xlabel('East [m]', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    axes[0].set_ylabel('North [m]', fontsize=10)
    plt.suptitle('Intentional Noise Injection: Trajectory Degradation vs LiDAR XY Noise',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'fig6_intentional_noise_trajectories.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_kf_vs_raw_lidar_sparse(results_by_rate, rate_labels):
    """KF vs raw LiDAR at progressively sparser update rates — shows KF benefit."""
    fig, axes = plt.subplots(2, len(rate_labels), figsize=(5*len(rate_labels), 9))
    for col, (res, label) in enumerate(zip(results_by_rate, rate_labels)):
        mm = compute_metrics(res)
        t  = res['t']
        ax = axes[0, col]
        ax.plot(res['px_gt'], res['py_gt'], 'k-', lw=2.0, label='Ground Truth')
        ax.plot(res['kf_states'][:,0], res['kf_states'][:,1], 'b-', lw=1.5, alpha=0.9,
                label=f"KF ({mm['kf_rmse']:.2f}m)")
        ax.plot(res['lio_px'], res['lio_py'], 'g-', lw=1.0, alpha=0.6,
                label=f"LiO ({mm['lio_rmse']:.2f}m)")
        ax.set_title(f'LiDAR @ {label}', fontsize=11, fontweight='bold')
        ax.set_xlabel('East [m]', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
        if col == 0: ax.set_ylabel('North [m]', fontsize=10)
        ax2 = axes[1, col]
        ax2.plot(t, mm['kf_pos_err'],  'b-',  lw=1.5, label=f"KF RMSE={mm['kf_rmse']:.2f}m")
        ax2.plot(t, mm['lio_pos_err'], 'g--', lw=1.2, alpha=0.8,
                 label=f"Raw LiO RMSE={mm['lio_rmse']:.2f}m")
        ax2.set_xlabel('Time [s]', fontsize=9)
        ax2.set_ylabel('Position Error [m]', fontsize=9)
        ax2.set_title(f'Error @ {label}', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.suptitle('KF vs Raw LiDAR Odometry: Benefit Grows with Sparser LiDAR Updates',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'fig7_kf_benefit_sparse_lidar.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_update_rate_vs_accuracy_and_compute(rate_vals, rate_labels,
                                              rmse_kf, rmse_lio, compute_vals):
    """LiDAR update rate vs both RMSE and compute cost."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(rate_labels))
    w = 0.35
    bars1 = axes[0].bar(x - w/2, rmse_kf,  w, color='#3498db', edgecolor='black', label='KF Fusion')
    bars2 = axes[0].bar(x + w/2, rmse_lio, w, color='#f39c12', edgecolor='black', label='Raw LiDAR Odometry')
    axes[0].set_xticks(x); axes[0].set_xticklabels(rate_labels, fontsize=10)
    axes[0].set_xlabel('LiDAR Update Rate', fontsize=11)
    axes[0].set_ylabel('Position RMSE [m]', fontsize=11)
    axes[0].set_title('Accuracy vs LiDAR Update Rate', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10); axes[0].grid(True, axis='y', alpha=0.3)
    for bar in list(bars1) + list(bars2):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                     f'{bar.get_height():.2f}m', ha='center', va='bottom', fontsize=9)
    axes[1].plot(rate_vals, compute_vals, 'rs-', lw=2, markersize=9)
    for rv, cv, lbl in zip(rate_vals, compute_vals, rate_labels):
        axes[1].annotate(f'{cv:.1f}ms ({lbl})', (rv, cv),
                         textcoords='offset points', xytext=(6, 4), fontsize=9)
    axes[1].set_xlabel('LiDAR Update Rate [Hz]', fontsize=11)
    axes[1].set_ylabel('Compute Time [ms / 120 s run]', fontsize=11)
    axes[1].set_title('Computational Cost vs LiDAR Update Rate', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('LiDAR Update Rate: Accuracy and Computational Efficiency Trade-off',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(plot_dir, 'fig8_update_rate_accuracy_compute.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')

if __name__ == '__main__':
    print("=" * 60)
    print("  Project 2: LiDAR-KF Vehicle Localization")
    print("=" * 60)

    # ── Baseline run ──────────────────────────────────────────────────────────
    print("\n[1] Running baseline simulation...")
    res = run_simulation()
    m   = compute_metrics(res)

    print(f"\n  KF   RMSE: {m['kf_rmse']:.3f} m  | Mean: {m['kf_mean']:.3f} m  | Max: {m['kf_max']:.3f} m")
    print(f"  DR   RMSE: {m['dr_rmse']:.3f} m  | Mean: {m['dr_mean']:.3f} m  | Max: {m['dr_max']:.3f} m")
    print(f"  LiO  RMSE: {m['lio_rmse']:.3f} m  | Mean: {m['lio_mean']:.3f} m  | Max: {m['lio_max']:.3f} m")
    print(f"  KF Heading RMSE: {m['kf_head_rmse']:.3f} deg")
    print(f"  Compute: {res['compute_time_ms']:.2f} ms")

    print("\n[2] Generating baseline plots...")
    plot_trajectory(res, m)
    plot_errors(res, m)
    plot_rmse_bar(m)
    plot_lidar_noise_characterization(res)
    plot_kalman_gain(res)

    # ── Sensitivity: LiDAR XY noise ───────────────────────────────────────────
    print("\n[3] Sensitivity sweep: LiDAR XY noise...")
    rmse_kf_xy, rmse_lio_xy = [], []
    for v in SWEEP_LIDAR_XY_NOISE:
        r = run_simulation(lidar_xy_noise=v)
        mm = compute_metrics(r)
        rmse_kf_xy.append(mm['kf_rmse']); rmse_lio_xy.append(mm['lio_rmse'])
        print(f"  σ_xy={v:.3f}m  → KF RMSE={mm['kf_rmse']:.3f}m  LiO RMSE={mm['lio_rmse']:.3f}m")
    plot_sensitivity_sweep('xy_noise', SWEEP_LIDAR_XY_NOISE,
                           rmse_kf_xy, rmse_lio_xy, '{:.2f}m', 'LiDAR XY Noise σ [m]')

    # ── Sensitivity: LiDAR heading noise ─────────────────────────────────────
    print("\n[4] Sensitivity sweep: LiDAR heading noise...")
    rmse_kf_psi, rmse_lio_psi = [], []
    for v in SWEEP_LIDAR_PSI_NOISE:
        r = run_simulation(lidar_psi_noise=v)
        mm = compute_metrics(r)
        rmse_kf_psi.append(mm['kf_rmse']); rmse_lio_psi.append(mm['lio_rmse'])
        print(f"  σ_psi={v:.4f}rad → KF RMSE={mm['kf_rmse']:.3f}m  LiO RMSE={mm['lio_rmse']:.3f}m")
    plot_sensitivity_sweep('psi_noise', SWEEP_LIDAR_PSI_NOISE,
                           rmse_kf_psi, rmse_lio_psi, '{:.3f}rad', 'LiDAR Heading Noise σ [rad]')

    # ── Sensitivity: drift coefficient ────────────────────────────────────────
    print("\n[5] Sensitivity sweep: drift coefficient...")
    rmse_kf_dr, rmse_lio_dr = [], []
    for v in SWEEP_LIDAR_DRIFT:
        r = run_simulation(drift_xy=v, drift_psi=v*0.5)
        mm = compute_metrics(r)
        rmse_kf_dr.append(mm['kf_rmse']); rmse_lio_dr.append(mm['lio_rmse'])
        print(f"  drift={v:.4f}  → KF RMSE={mm['kf_rmse']:.3f}m  LiO RMSE={mm['lio_rmse']:.3f}m")
    plot_sensitivity_sweep('drift', SWEEP_LIDAR_DRIFT,
                           rmse_kf_dr, rmse_lio_dr, '{:.4f}', 'Drift Coefficient')

    # ── Accuracy vs compute (time step) ──────────────────────────────────────
    print("\n[6] Accuracy vs compute trade-off (varying Δt)...")
    rmse_kf_dt, compute_dt = [], []
    for v in SWEEP_DT:
        r  = run_simulation(dt=v)
        mm = compute_metrics(r)
        rmse_kf_dt.append(mm['kf_rmse'])
        compute_dt.append(r['compute_time_ms'])
        print(f"  Δt={v:.2f}s → KF RMSE={mm['kf_rmse']:.3f}m  Compute={r['compute_time_ms']:.2f}ms")
    plot_accuracy_vs_compute(SWEEP_DT, rmse_kf_dt, compute_dt)

    # ── GAP 1: Intentional noise injection — side-by-side trajectory degradation ─
    print("\n[7] Intentional noise injection (trajectory degradation)...")
    noisy_sweep_vals  = [0.02, 0.08, 0.50]   # clean → degraded → very noisy
    noisy_sweep_res   = []
    for v in noisy_sweep_vals:
        r = run_simulation(lidar_xy_noise=v)
        noisy_sweep_res.append(r)
        mm = compute_metrics(r)
        print(f"  σ_xy={v:.2f}m → KF RMSE={mm['kf_rmse']:.3f}m  LiO RMSE={mm['lio_rmse']:.3f}m")
    plot_intentional_noise_trajectories(noisy_sweep_res, noisy_sweep_vals)

    # ── GAP 2: KF vs raw LiDAR at sparse update rates ────────────────────────
    print("\n[8] KF benefit with sparse LiDAR updates...")
    sparse_every  = [1, 5, 10, 20]           # every N steps → 10, 2, 1, 0.5 Hz
    sparse_labels = ['10 Hz', '2 Hz', '1 Hz', '0.5 Hz']
    sparse_res    = []
    for every in sparse_every:
        r  = run_simulation(lidar_every=every)
        mm = compute_metrics(r)
        sparse_res.append(r)
        gap = mm['lio_rmse'] - mm['kf_rmse']
        print(f"  LiDAR every {every:2d} steps ({sparse_labels[sparse_every.index(every)]:5s})"
              f" → KF={mm['kf_rmse']:.3f}m  LiO={mm['lio_rmse']:.3f}m  KF_gain={gap:.3f}m")
    plot_kf_vs_raw_lidar_sparse(sparse_res, sparse_labels)

    # ── GAP 3: Update rate vs accuracy AND compute ────────────────────────────
    print("\n[9] LiDAR update rate: accuracy and compute trade-off...")
    rate_hz    = [10.0, 2.0, 1.0, 0.5]
    rate_rmse_kf, rate_rmse_lio, rate_compute = [], [], []
    for every, r, lbl in zip(sparse_every, sparse_res, sparse_labels):
        mm = compute_metrics(r)
        rate_rmse_kf.append(mm['kf_rmse'])
        rate_rmse_lio.append(mm['lio_rmse'])
        rate_compute.append(r['compute_time_ms'])
    plot_update_rate_vs_accuracy_and_compute(
        rate_hz, sparse_labels, rate_rmse_kf, rate_rmse_lio, rate_compute)

    # ── Save all stats ────────────────────────────────────────────────────────
    all_stats = {
        'baseline': {k: float(v) if not isinstance(v, np.ndarray) else None
                     for k, v in m.items() if not isinstance(v, np.ndarray)},
        'compute_time_ms': float(res['compute_time_ms']),
        'sensitivity_xy_noise':  {'values': SWEEP_LIDAR_XY_NOISE,  'kf_rmse': rmse_kf_xy,  'lio_rmse': rmse_lio_xy},
        'sensitivity_psi_noise': {'values': SWEEP_LIDAR_PSI_NOISE, 'kf_rmse': rmse_kf_psi, 'lio_rmse': rmse_lio_psi},
        'sensitivity_drift':     {'values': SWEEP_LIDAR_DRIFT,     'kf_rmse': rmse_kf_dr,  'lio_rmse': rmse_lio_dr},
        'accuracy_vs_compute':   {'dt': SWEEP_DT, 'kf_rmse': rmse_kf_dt, 'compute_ms': compute_dt},
    }
    with open(os.path.join(output_dir, 'stats_p2.json'), 'w') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*60}")
    print("  All plots saved to ./plots_p2/")
    print(f"{'='*60}")
