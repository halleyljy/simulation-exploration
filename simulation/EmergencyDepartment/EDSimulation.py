"""
Simulate a patient experience when visiting the emergency department
"""

from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import Any, Generator

import simpy
import numpy as np
from pydantic import BaseModel, Field, ValidationError

import common.genericfunction as gf

# ------------------ input validation ------------------
class BranchProbabilityMap(BaseModel):
    """
    Used to validate simulation branch probabilities in input

    Attributes:
        fast_track_probability (float): probability of patient arriving being in fast track
        fast_track_lab_probability (float): probability of patients in fast track that requires lab tests
        main_ed_lab_probability (float): probability of patients in main ED that requires lab tests
        main_ed_admission_probability (float): probability of patients in main ed that has to be admitted
    """
    fast_track_probability: float = Field(gt=0)
    fast_track_lab_probability: float = Field(gt=0)
    main_ed_lab_probability: float = Field(gt=0)
    main_ed_admission_probability: float = Field(gt=0)

class Resources(BaseModel):
    """
    Used to validate simulation resources in input

    Attributes:
        fast_track_doctor (int): number of doctors in fast track
        fast_track_nurse (int): number of nurses in fast track
        main_ed_doctor (int): number of doctors in main ed
        main_ed_nurse (int): number of nurses in main ed
        ed_beds (int): number of beds in emergency department
        ed_labs (int): number of labs available for testing
    """
    fast_track_doctor: int = Field(gt=0)
    fast_track_nurse: int = Field(gt=0)
    main_ed_doctor: int = Field(gt=0)
    main_ed_nurse: int = Field(gt=0)
    ed_beds: int = Field(gt=0)
    ed_labs: int = Field(gt=0)

class DelayTimeData(BaseModel):
    """ Used to validate time delay value in inputs """
    min_value: float = Field(gt=0)
    mean: float = Field(gt=0)

class PatientTrackDelayTime(BaseModel):
    """ Used to validate patient track input """
    main_ed: DelayTimeData
    fast_track: DelayTimeData

class PatientAdmissionDelayTime(BaseModel):
    """ Used to validate patient admission input """
    main_ed: DelayTimeData

class TimeParameter(BaseModel):
    """
    Used to validate different service types
    
    attribute:
        patient_arrival_rate_per_hour(float): average patient arrivals per hour
        service_delay_minutes: time parameters for patient visiting doctor in minutes
        lab_delay_minutes: time parameters for patient lab test in minutes
        transfer_delay_minutes: time parameters for patient waiting for transfer to other deparment
    """
    patient_arrival_rate_per_hour: float = Field(gt=0)
    service_delay_minutes: PatientTrackDelayTime
    lab_delay_minutes: PatientTrackDelayTime
    transfer_delay_minutes: PatientAdmissionDelayTime

class SimulationInput(BaseModel):
    """ Used to validate simulation input read from setting file """
    output_directory: str = Field(min_length=1)
    scenario_name: str = Field(min_length=1)
    total_replications: int = Field(gt=0)
    simulation_days: int = Field(gt=0)
    seed_value: int = Field(gt=0)
    stats_collection_interval_minutes: int = Field(gt=0)
    branch_probability_map: BranchProbabilityMap
    resources: Resources
    time_parameter: TimeParameter

def check_scenario(scenario: dict, logger:Any) -> bool:
    """ Check settings given to see if user input is wrong """
    logger.info('Enter check scenario function of ED simulation')

    if not isinstance(scenario, dict):
        logger.error('Input is not given properly, please review the settings file.')
        return False
    
    try:
        SimulationInput(**scenario)
        return True
    except ValidationError as e:
        for error in e.errors():
            input_name = '-'.join(error['loc'])
            input_issue = error['type']
            input_fix = error['msg']
            logger.error(f'Input with issue: {input_name}, Issue found: {input_issue}, Note: {input_fix}')
        return False


# ------------------ Helper variables ------------------
HOURS_IN_MINUTES = 60.0
HOURS_IN_A_DAY = 24

class Constant(Enum):
    """
    Records all constants that are used in the emergency department simulation
    """

    EMPTY = ''
    MAIN_TRACK = 'main_ed'
    FAST_TRACK = 'fast_track'
    DISCHARGED = 'Discharged'
    ADMITTED = 'Admitted'
    TOTAL_REPLICATIONS = 'total_replications'


# ------------------ Helper function ------------------
def get_delay_time_information(delay_times: TimeParameter, patient_track: str, service_type: str) -> tuple[float, float]:
    """ Get min and mean delay time based on service type and track for patient """

    lookup_table: dict[str, Any] = delay_times.__dict__
    delay_times: PatientTrackDelayTime = lookup_table[service_type]
    filtered_table: dict[str, Any] = delay_times.__dict__
    final_table: DelayTimeData = filtered_table[patient_track]
    min_delay_minutes, mean_delay_minutes = (final_table.min_value, final_table.mean)
    return  min_delay_minutes, mean_delay_minutes

def get_delay_minutes(min_delay_minutes: float, mean_delay_minutes: float) -> float:
    """ return delay time """
    return min_delay_minutes + np.random.exponential(mean_delay_minutes - min_delay_minutes)


# ------------------ Stats collector classes ------------------
@dataclass
class SimulationResources:
    """ Created for easy access of simulation resource """
    def __init__(self, env: simpy.Environment, params: Resources):
        for attribute in type(params).model_fields:
            capacity:int = getattr(params, attribute)
            setattr(self, attribute, simpy.Resource(env, capacity=capacity))

@dataclass(kw_only=True)
class Patient:
    """
    Records all statistics for each patients entering the emergency department

    This class models all statistics for every patient entering the emergency department to output to csv file

    Attributes:
        replication (int): Record replication number of the simulation
        patient_id (int): Record the id of the patient
        route (str): Records track (Main ED/ Fast track) the patient enters
        arrival_time (float): Record time when patient enter the hospital
        start_service (float): Record time when patient is first seen by the doctor
        end_service (float): Record time when patient is discharged or transferred out of ED
        disposition (str): Record if patient is discharged/ admitted
        boarding_start (float): Record time when patient completed all treatment is only waiting to be transferred to another department
        boarding_end (float): Record time when patient is transferred to another department for admission
    """
    replication_number: int
    patient_id: int
    route: str = np.nan
    arrival_time: float
    start_service: float = np.nan
    end_service: float = np.nan
    disposition: str = Constant.EMPTY.value
    boarding_start: float = np.nan
    boarding_end: float = np.nan

@dataclass
class Queue:
    """
    Records statistics for resources used in the hospital

    This class collects and records resources used in the simulation time specified to output to csv file
    This class uses the resource dictionary to determine number of the resources used

    Attributes:
        replication (int): Record current replication number of the simulation
        time_now (float): Record current simulation time
        beds_in_use (int): Record number of ED beds that are currently used
        bed_queue_length (int): Record number of patients waiting to use the ED beds
        main_doctor_queue_length (int): Record number of patient queueing for main track doctor
        main_nurse_queue_length (int): Record number of patient queueing for main track nurse
        fast_doctor_queue_length (int): Record number of patient queueing for fast track doctor
        fast_nurse_queue_length (int): Record number of patient queueing for fast track nurse
    """
    # required inputs for initialisation
    replication_number: int
    time_now: float
    resources: InitVar[SimulationResources] # will not be stored

    # fields calculated automatically (excluded from __init__)
    beds_in_use: int = field(init=False)
    bed_queue_length: int = field(init=False)
    ed_labs_queue: int = field(init=False)
    main_doctor_queue_length: int = field(init=False)
    main_nurse_queue_length: int = field(init=False)
    fast_doctor_queue_length: int = field(init=False)
    fast_nurse_queue_length: int = field(init=False)

    def __post_init__(self, resources: dict[str, Any]):
        self.beds_in_use = resources.ed_beds.count
        self.bed_queue_length = len(resources.ed_beds.queue)
        self.ed_labs_queue = len(resources.ed_labs.queue)
        self.main_doctor_queue_length = len(resources.main_ed_doctor.queue)
        self.main_nurse_queue_length = len(resources.main_ed_nurse.queue)
        self.fast_doctor_queue_length = len(resources.fast_track_doctor.queue)
        self.fast_nurse_queue_length = len(resources.fast_track_nurse.queue)

@dataclass(kw_only=True)
class EDStats:
    """
    Records the summary of the simulation statistics

    This class collects and records summary of the simulation result to output to csv file
    This class uses list of various data to find mean and 90th percentile

    Attributes:
        replication (int): Record replication number of the simulation
        avg_wait_min (float): Average wait time from patient arrival to seen by doctor
        p90_wait_min (float): 90th Percentile wait time from patient arrival to seen by doctor
        avg_los_min (float): Average length of stay from patient arrival till discharge or transfer
        p90_los_min (float): 90th Percentile length of stay from patient arrival till discharge or transfer
        num_discharged (int): Record number of patient discharged from hospital
        num_admitted (int): Record number of patient transferred out of the emergency department for admission
        avg_boarding_min (float): Average boarding time from treatment completed till transfer for admission
    """
    replication_number: int
    avg_wait_min: float = field(init=False)
    p90_wait_min: float = field(init=False)
    avg_los_min: float = field(init=False)
    p90_los_min: float = field(init=False)
    num_discharged: int
    num_admitted: int
    avg_boarding_min: float = field(init=False)

    wait_time_array: InitVar[list[float]] # will not be stored
    los_array: InitVar[list[float]] # will not be stored
    boarding_durations: InitVar[list[float]] # will not be stored

    def __post_init__(self, wait_time_array, los_array, boarding_durations) -> None:
        self.avg_wait_min = np.mean(wait_time_array) if wait_time_array.size > 0 else 0.0
        self.p90_wait_min = np.percentile(wait_time_array, 90) if wait_time_array.size > 0 else 0.0
        self.avg_los_min = np.mean(los_array) if los_array.size > 0 else 0.0
        self.p90_los_min = np.percentile(los_array, 90) if los_array.size > 0 else 0.0
        self.avg_boarding_min = np.mean(boarding_durations) if boarding_durations.size > 0 else 0.0

@dataclass
class Collector:
    """
    Records all statistics to output

    This class models all statistics that will be created in a csv file.
    It includes module to summarise all the data within the same replication

    Attributes:
        num_discharged (int): Number of patients that are discharged from hospital
        num_admitted (int): Number of patients that is admitted to hospital
        patient_stats (list): Holds every instances of patient class
        queue_samples (list): Holds every instances of queue class
    """
    replication_number: int
    num_discharged: int = 0
    num_admitted: int = 0
    patient_stats: list[Any] = field(default_factory=list)
    queue_samples: list[Any] = field(default_factory=list)

    def summarise(self, logger: Any) -> list[dict]:
        """ Summarise and return all statistics to output """
        logger.info('get info from statistics')
        list_of_patient_data: list[Patient] = self.patient_stats
        list_of_queue_data: list[dict] = self.queue_samples
        
        logger.info('convert data into type that is easy to convert to csv and specify name of the output file')
        patient_records: dict[list[dict]] = {'patient_records': [patient_data.__dict__ for patient_data in list_of_patient_data]}
        queue_records: dict[list[dict]] = {'queue_records': list_of_queue_data}

        logger.info('retrieve information that needs to be processed to get summary of ED simulation')
        patient_data_summary: dict[list] = gf.list_of_instance_to_dict_of_lists(list_of_patient_data)

        logger.info('get time passed between events')
        wait_time_array: np.array = gf.get_difference_between_two_lists(patient_data_summary['start_service'], patient_data_summary['arrival_time'])
        los_array: np.array = gf.get_difference_between_two_lists(patient_data_summary['end_service'], patient_data_summary['arrival_time'])
        boarding_durations: np.array = gf.get_difference_between_two_lists(patient_data_summary['boarding_end'], patient_data_summary['boarding_start'])

        # summarise data from one replication into a row
        logger.info(f'Summarise simulation result for repetition: {self.replication_number}')
        simulation_result = EDStats(
            replication_number = self.replication_number,
            num_discharged = self.num_discharged,
            num_admitted = self.num_admitted,
            wait_time_array = wait_time_array,
            los_array = los_array,
            boarding_durations = boarding_durations
        )
        simulation_result_record: dict = {'ed_summary': [simulation_result.__dict__]}

        return [patient_records, queue_records, simulation_result_record]

@dataclass
class EmergencyDepartment:
    """ Created to hold lookup information throughout simulation """
    time_parameter: TimeParameter
    env: simpy.Environment
    collector: Collector
    resources: SimulationResources
    probability_map: BranchProbabilityMap


# ------------------ emergency department simulation function ------------------
def patient_visit_doctor(logger: Any, ed: EmergencyDepartment, patient: Patient, doctor: simpy.Resource, nurse:simpy.Resource) -> Generator[Any, None, None]:
    """ process patient's visit to the doctor """
    with doctor.request() as rd, nurse.request() as rn:
        yield rd & rn
        logger.debug(f'Patient: {patient.patient_id} is seen by the doctor at simulation time: {ed.env.now}')
        if np.isnan(patient.start_service):
            patient.start_service = ed.env.now
        
        min_delay_minutes, mean_delay_minutes = get_delay_time_information(ed.time_parameter, patient.route, 'service_delay_minutes')
        yield ed.env.timeout(get_delay_minutes(min_delay_minutes, mean_delay_minutes))
        logger.debug(f'Patient: {patient.patient_id} leaves doctor office at simulation time: {ed.env.now}')

def patient_lab_test(logger: Any, ed: EmergencyDepartment, patient: Patient, lab: simpy.Resource, nurse: simpy.Resource) -> Generator[Any, None, None]:
    """ process patient's lab test """
    with lab.request() as rl, nurse.request() as rn:
        yield rl & rn
        logger.debug(f'Patient: {patient.patient_id} starts lab test at simulation time: {ed.env.now}')
        
        min_delay_minutes, mean_delay_minutes = get_delay_time_information(ed.time_parameter, patient.route, 'lab_delay_minutes')
        yield ed.env.timeout(get_delay_minutes(min_delay_minutes, mean_delay_minutes))
        logger.debug(f'Patient: {patient.patient_id} completed lab test at simulation time: {ed.env.now}')

def patient_transfer(logger: Any, ed: EmergencyDepartment, patient: Patient) -> Generator[Any, None, None]:
    """ process patient's transfer to another apartment """
    patient.boarding_start = ed.env.now
    min_delay_minutes, mean_delay_minutes = get_delay_time_information(ed.time_parameter, patient.route, 'transfer_delay_minutes')
    yield ed.env.timeout(get_delay_minutes(min_delay_minutes, mean_delay_minutes))
    patient.boarding_end = ed.env.now
    logger.debug(f'Patient: {patient.patient_id} transferred to another department at simulation time: {ed.env.now}')

def patient_process(logger: Any, ed: EmergencyDepartment, patient_id: int) -> Generator[Any, None, None]:
    """ Simulate the entire process of patients coming into the emergency department """

    logger.info(f'Enter patient process function for patient: {patient_id}')
    t_arr: float = ed.env.now
    patient = Patient(
        replication_number = ed.collector.replication_number,
        patient_id = patient_id,
        arrival_time = t_arr
    )
    ed.collector.patient_stats.append(patient)

    with ed.resources.ed_beds.request() as bedreq:
        yield bedreq  # wait until a bed is available
        logger.debug(f'Patient: {patient_id} obtains a bed at simulation time: {ed.env.now}')
        
        # Decide route
        if np.random.rand() < ed.probability_map.fast_track_probability:
            logger.info(f'Patient: {patient_id} in fast track')
            # Fast Track: needs both doctor + nurse
            patient.route = Constant.FAST_TRACK.value

            yield ed.env.process(patient_visit_doctor(logger, ed, patient, ed.resources.fast_track_doctor, ed.resources.fast_track_nurse))
                
            # lab
            if np.random.rand() < ed.probability_map.fast_track_lab_probability:
                logger.info(f'Patient: {patient_id} requires lab test')
                # lab test: needs nurse
                yield ed.env.process(patient_lab_test(logger, ed, patient, ed.resources.ed_labs, ed.resources.fast_track_nurse))

                # review lab test result: needs doctor + nurse
                yield ed.env.process(patient_visit_doctor(logger, ed, patient, ed.resources.fast_track_doctor, ed.resources.fast_track_nurse))
            
            logger.info(f'Patient: {patient_id} discharged at simulation time: {ed.env.now}')
            patient.end_service = ed.env.now
            patient.disposition = Constant.DISCHARGED.value
            ed.collector.num_discharged += 1
            return
        
        # Main ED: needs both doctor + nurse
        patient.route = Constant.MAIN_TRACK.value
        logger.info(f'Patient: {patient_id} in main ED')

        yield ed.env.process(patient_visit_doctor(logger, ed, patient, ed.resources.main_ed_doctor, ed.resources.main_ed_nurse))
        
        if np.random.rand() < ed.probability_map.main_ed_lab_probability:
            logger.debug(f'Patient: {patient_id} requires lab test')
            # lab test: needs nurse
            yield ed.env.process(patient_lab_test(logger, ed, patient, ed.resources.ed_labs, ed.resources.main_ed_nurse))
            
            # review lab test result: needs doctor + nurse
            yield ed.env.process(patient_visit_doctor(logger, ed, patient, ed.resources.main_ed_doctor, ed.resources.main_ed_nurse))
        
        if not np.random.rand() < ed.probability_map.main_ed_admission_probability:
            logger.info(f'Patient: {patient_id} discharged at simulation time: {ed.env.now}')
            patient.end_service = ed.env.now
            patient.disposition = Constant.DISCHARGED.value
            ed.collector.num_discharged += 1
            return
        
        logger.info(f'Patient: {patient_id} is waiting to be transffered to another department')
        yield ed.env.process(patient_transfer(logger, ed, patient))
        
        patient.end_service = ed.env.now
        patient.disposition = Constant.ADMITTED.value
        ed.collector.num_admitted += 1
        logger.info(f'Patient: {patient_id} admitted to hospital at simulation time: {ed.env.now}')

def patient_arrivals(logger: Any, ed: EmergencyDepartment) -> Generator[Any, None, None]:
    """" Simulate patient arrival to clinic so that patient go through ed process """
    logger.info('Enter patient arrival function')
    patient_count: int = 0
    lam: float = HOURS_IN_MINUTES / ed.time_parameter.patient_arrival_rate_per_hour
    while True:
        time_between_each_arrival: float = np.random.exponential(lam)
        yield ed.env.timeout(time_between_each_arrival)
        patient_count += 1
        logger.debug(f'Simulation time: {ed.env.now}, patient number: {patient_count} goes through ED process')
        ed.env.process(patient_process(logger, ed, patient_count))


# ------------------ Monitor queue ------------------
def monitor_queues(logger: Any, ed: EmergencyDepartment, time_interval: int) -> Generator[Any, None, None]:
    """ Track resources used periodically during the simulation """
    logger.info('Enter periodic monitor queue function')
    while True:
        logger.info(f'Check queue situation at simulation time: {ed.env.now}, next time to collect is: {ed.env.now + time_interval}')
        logger.debug(f'Record stats for queue with {ed.collector.replication_number}')
        queue_sample = Queue(ed.collector.replication_number, ed.env.now, ed.resources)
        queue_dict = queue_sample.__dict__
        ed.collector.queue_samples.append(queue_dict)
        
        queue_str = str(queue_dict)
        logger.debug(f'Simulation time: {ed.env.now}, collected queue stats: ' + queue_str)
        yield ed.env.timeout(time_interval)


# ------------------ main function that runs simulation ------------------
def run_one_replication(params: dict, logger: Any, replication_number: int, seed: int) -> Collector:
    """ Initialise and run simulation """
    logger.info('Run repetition part 1: Initiliase simulation')
    np.random.seed(seed)
    setting = SimulationInput(**params)
    sim_minutes = int(setting.simulation_days * HOURS_IN_A_DAY * HOURS_IN_MINUTES)

    time_parameter = setting.time_parameter
    logger.debug(f'Get delay time lookup table: {time_parameter}')

    
    env = simpy.Environment()
    collector = Collector(replication_number)
    resources = SimulationResources(env, setting.resources)
    ed = EmergencyDepartment(time_parameter, env, collector, resources, setting.branch_probability_map)

    time_interval: int = setting.stats_collection_interval_minutes
    
    # processes
    logger.info('Run repetition part 2: Run simulation')
    env.process(patient_arrivals(logger, ed))
    env.process(monitor_queues(logger, ed, time_interval))
    env.run(until=sim_minutes)

    logger.info('Run repetition part 3: Summarise simulation result')
    output_statistic: list[dict] = collector.summarise(logger)
    return output_statistic
