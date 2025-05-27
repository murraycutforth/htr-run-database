import csv
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from src.output_utils.main_assemble_pressure_trace_matrix import read_all_pressure_files, get_average_pressure_trace, \
    load_xi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Configuration
@dataclass
class Config:
    MAX_LASER_DELAY: int = 21  # Based on the zone1 time step and the laser delay iterations
    MAX_TIME: int = 40  # tref is 1.1375e-5 [s] so 60 -> 700us
    VALID_LASER_DELAYS: Tuple[int] = (1000, 2000, 3000, 4000, 5000, 6000)
    IGNITION_THRESHOLD: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process pressure files.")
    parser.add_argument('--run-dir', type=str, required=True, help='Path to directory where runs are stored')
    return parser.parse_args()


def load_run_data(solution_dir: Path) -> Optional[Tuple[int, pd.DataFrame]]:
    """Load run data from a solution directory."""
    try:
        run_id = int(solution_dir.parent.name)
        if not solution_dir.exists():
            raise FileNotFoundError(f"Solution directory {solution_dir} does not exist.")
        
        probe_ind_to_df = read_all_pressure_files(solution_dir)
        avg_pressure_trace = get_average_pressure_trace(probe_ind_to_df)
        return run_id, avg_pressure_trace
    except Exception as e:
        logger.error(f"Failed to load run {solution_dir.parent.name}: {str(e)}")
        return None


def validate_laser_delay(xi: np.ndarray) -> bool:
    """Validate laser delay parameters."""
    laser_delay_iterations = int(xi[16])
    return (laser_delay_iterations % 1000 == 0 and 
            laser_delay_iterations >= 1000 and 
            laser_delay_iterations in Config.VALID_LASER_DELAYS)


def process_time_shift(run_id: int, df: pd.DataFrame, xi: np.ndarray) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
    """Process time shift for a single run."""
    if not validate_laser_delay(xi):
        logger.warning(f"Invalid laser delay for run {run_id}")
        return None

    laser_delay_iterations = int(xi[16])
    start_ind = (laser_delay_iterations + 1000) // 10
    
    if len(df) < start_ind:
        logger.warning(f"Run {run_id} is too short to apply laser time shift")
        return None

    df = df.iloc[start_ind:].copy()
    start_time = df['Time'].iloc[0]

    if start_time > Config.MAX_LASER_DELAY:
        logger.warning(f"Start time {start_time} exceeds MAX_LASER_DELAY")
        return None

    df['Time_shifted'] = df['Time'] - start_time

    if df['Time_shifted'].max() < Config.MAX_TIME - 1:
        logger.warning(f"Run {run_id} is too short after time shift")
        return None

    return df, xi


def calculate_pressures(run_id_to_df: Dict[int, pd.DataFrame]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Calculate pressure values and ignition status."""
    pressures = [df['Avg_Pressure'].values for df in run_id_to_df.values()]
    avg_p0 = np.mean([p[0] for p in pressures])
    pressures_normalized = [p - avg_p0 for p in pressures]
    chis = np.array([p[-1] > Config.IGNITION_THRESHOLD for p in pressures_normalized])
    return pressures_normalized, chis


def save_results(output_path: Path, xis: np.ndarray, chis: np.ndarray) -> None:
    """Save results to NPZ file."""
    if output_path.exists():
        raise FileExistsError(f"Output file {output_path} already exists")
    
    np.savez_compressed(output_path, xis=xis, chis=chis)
    logger.info(f'Saved results to {output_path}')


def plot_ignitions(xis: np.ndarray, chis: np.ndarray) -> None:
    """Plot ignition status."""
    all_xi = np.array(xis)
    xs = all_xi[:, 2].astype(float)
    zs = all_xi[:, 4].astype(float)
    fig, ax = plt.subplots(dpi=200)
    colors = ['red' if chi else 'blue' for chi in chis]
    ax.scatter(xs, zs, s=10, c=colors, alpha=0.5)

    ax.set_xlabel('Radial Distance [mm]')
    ax.set_ylabel('Streamwise Distance [mm]')
    ax.set_aspect('equal')

    plt.show()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")

    solution_dirs = list(run_dir.glob('*/solution'))
    logger.info(f'Found {len(solution_dirs)} solution directories in {run_dir}')

    # Load run data
    run_id_to_df = {}
    for solution_dir in tqdm(solution_dirs, desc="Loading runs"):
        result = load_run_data(solution_dir)
        if result:
            run_id, df = result
            run_id_to_df[run_id] = df

    # Process time shifts
    xis = [load_xi(run_id) for run_id in run_id_to_df.keys()]
    processed_data = {}
    processed_xis = []
    
    for run_id, (df, xi) in zip(run_id_to_df.keys(), zip(run_id_to_df.values(), xis)):
        result = process_time_shift(run_id, df, xi)
        if result:
            processed_df, processed_xi = result
            processed_data[run_id] = processed_df
            processed_xis.append(processed_xi)

    logger.info(f'After laser time shift, we have {len(processed_data)} runs')

    # Calculate pressures and save results
    pressures, chis = calculate_pressures(processed_data)
    plot_ignitions(np.array(processed_xis), chis)
    
    output_path = Path(f'ignitions_{run_dir.stem}.npz')
    save_results(output_path, np.array(processed_xis), chis)


if __name__ == "__main__":
    main()