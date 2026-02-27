# %% [markdown]
# Simulation functions for ED model.
# - run_one_replication(params, seed)
# - run_scenario(params, scenario_name, outdir)
# 
# Outputs are saved into outdir:
# - ed_summary.csv (one row per replication)
# - wait_samples.csv (each patient wait times)
# - los_samples.csv (each patient LOS)
# - queue_samples.csv (time, bed_queue_len, main_doc_queue_len, main_nurse_queue_len, fast_queue_len, beds_in_use)
# - sim_settings.json

# %%
import logging
from multiprocessing import Pool
from pathlib import Path

import common.genericfunction as gf
import simulation.EmergencyDepartment.EDSimulation as ed

# ------------------ run simulation ------------------
def main() -> None:
    setting_file_name:Path = Path.cwd() / 'ed_sim_setting.json'
    setting: dict = gf.get_parameter_from_file(setting_file_name)
    directory: str = setting.get(gf.Constant.DIRECTORY.value, '')
    if not directory:
        directory = Path.cwd()
    log_file_name = Path(directory) / 'ED Simulation Log.log'
    log_file_name.touch(exist_ok=True)
    logger = gf.get_logger('main_logger', log_file_name, logging.INFO)
    
    if ed.check_scenario(setting, logger):
        logger.info('Safely pass all ED simulation scenario check')
        path = Path(directory) / setting[gf.Constant.SCENARIO.value]
        path.mkdir(exist_ok=True)
        rep = int(setting[ed.Constant.TOTAL_REPLICATIONS.value])
        seed_base = int(setting[gf.Constant.SEED.value])
        logger.debug(f'Obtained total repetition number:{rep} and seed data:{seed_base} from simulation parameters')
        
        use_multiprocessing: bool = setting[gf.Constant.MULTIPROCESSING.value]
        simulation_result_list = []
        if use_multiprocessing:
            with Pool(processes = 4) as pool:
                logger.info(f'Running scenario using multiprocessing')
                replication_seeds = [(setting, logger, rep_count + 1, seed_base + rep_count) for rep_count in range(rep)]
                # run scenario based on the input given
                simulation_result_list = pool.starmap(ed.run_one_replication, replication_seeds)
                logger.info(f'Completed scenarios for simulation')
        else:
            for rep_count in range(rep):
                logger.info(f'Running scenario for simulation repetition count: {rep_count + 1}')
                seed: int = seed_base + rep_count
                # run scenario based on the input given
                stats: list[dict] = ed.run_one_replication(setting, logger, rep_count + 1, seed)
                logger.info(f'Completed scenario for simulation repetition count: {rep_count + 1}')
                simulation_result_list.append(stats)
        
        # summarise data
        logger.info('Aggregate data collected')
        organised_data: dict[str, list] = gf.aggregate_all_replications(simulation_result_list)
        
        # write CSVs
        logger.info(f'Create output CSV files at directory: {path}')
        for filename, simulation_result in organised_data.items():
            logger.debug(f'Create csv for {filename}')
            filename += '.csv'
            gf.save_to_csv(simulation_result, path / filename)
            logger.debug(f'Successfully created csv for {filename}')
        
        # simulation name, parameter and timestamp for future reference
        logger.info(f'Create parameters file at directory: {path}')
        gf.save_parameter_file(setting, path / 'ed_simulation_setting.json')

if __name__ == '__main__':
    main()
