import model_pb2
import numpy as np
import google.protobuf.text_format
import ConfigParser
import logging
import pickle
import random
import os
import matplotlib
import matplotlib.pyplot as plt
from utils import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

config = ConfigParser.ConfigParser()
config.read('sim.conf')

N_STREAMS = int(config.get('sim', 'n_streams'))
SIM_LEN = int(config.get('sim', 'sim_len'))
SCHEDULE_METRIC = config.get('sim', 'schedule_metric') # fairness/completion

MACHINE_CORE = int(config.get('machine', 'n_core'))
MACHINE_MEMORY = float(config.get('machine', 'memory'))
MACHINE_COMPUTE = float(config.get('machine', 'flop'))
N_MACHINE = int(config.get('machine', 'n_machine'))

VIDEO_LIST = config.get('trace', 'video_list')
GREEDY_TRACE = config.get('trace', 'greedy_trace')
GREEDY_THRESH = float(config.get('trace', 'thresh'))
DICTIONARY_PATH = config.get('trace', 'dictionary')
START_FID_PATH = config.get('trace', 'sim_start_fid')

SP_TRACE_LOG = config.get('specialize', 'trace_log')
SP_CDF_N = int(config.get('specialize', 'N'))
SP_CDF_P = float(config.get('specialize', 'P'))
SP_MODEL_CONFIG = config.get('specialize', 'resource_file')
INTERM_MEM_REQ = float(config.get('specialize', 'interm_mem_req'))

class Model:
    def __init__(self, stream_id, model_config, create_ts):
        self.stream_id = stream_id  # -1: non-sp
        self.creat_ts = create_ts
        self.model_config = model_config 

class Job:
    def __init__(self, process_id, stream_id, frame_id, emit_ts, start_ts, end_ts, job_config):
        self.process_id = process_id
        self.stream_id = stream_id
        self.frame_id = frame_id
        self.emit_ts = emit_ts
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.config = job_config
        self.config['job_util'] = self.get_job_util()

    def __str__(self):
        return 'process_id:' + str(self.process_id) + ' stream:' + str(self.stream_id) + ' frame_id:' + str(self.frame_id) + ' emit_ts:' + str(self.emit_ts) +  ' start_ts:' + str(self.start_ts) + ' end_ts:' + str(self.end_ts)

    def get_job_util(self):
        return self.config['compute_flop']/((self.config['cpu_comp_lat']/1000.) * 1.0)

class Machine:

    def __init__(self, machine_id):
        self.machine_id = machine_id
        
        self.MEM_LIM = MACHINE_MEMORY
        self.CPU_LIM = MACHINE_COMPUTE

        self.mem_util = 0 
        self.core = MACHINE_CORE
        self.cpu_util = 0
  
        self.models = [] 
        self.jobs = [] 
        self.cpu_util_log = []  
 
    def __str__(self):
        return 'machine_id:' + str(self.machine_id) + ' mem_util (%):' + str((self.mem_util* 100)/self.MEM_LIM) + ' cpu_util (%):' + str((self.cpu_util * 100)/self.CPU_LIM)


    def does_model_fit(self, model_config):

        if self.mem_util + model_config['mem_req']  + model_config['mem_interm_req'] > self.MEM_LIM:

            return False

        return True

    def preload_model(self, stream_id, model_config):
        
        if self.does_model_fit(model_config): 

            self.models += [Model(stream_id, model_config, 0)]
            self.mem_util += model_config['mem_req']

            return True

        return False 

    def get_status(self):

        m_status = {'mem_util_byte': self.mem_util, 'mem_util_prec': self.mem_util/(self.MEM_LIM * 1.0), 'cpu_util_flop': self.cpu_util, 'cpu_util_prec': self.cpu_util/(self.CPU_LIM * 1.0), 'mem_lim': self.MEM_LIM, 'cpu_lim': self.CPU_LIM}

        return m_status

    def does_job_fit(self, job):
       
        # check if adding this job exceeds mem capacity
        if self.mem_util + job.config['mem_interm_req'] > self.MEM_LIM :
            return False

        # check if adding this job exceeds cpu util
        if self.cpu_util + job.config['job_util'] > self.CPU_LIM:
            return False

        return True 

    def does_job_model_fit(self, job):

        # check if adding this job exceeds mem capacity
        if self.mem_util + job.config['mem_interm_req'] + job.config['mem_req'] > self.MEM_LIM :
            return False

        # check if adding this job exceeds cpu util
        if self.cpu_util + job.config['job_util'] > self.CPU_LIM:
            return False

        return True 
        

    def run(self, scheme, job, cur_ts):

        if scheme == 'nsp': 
            if self.does_job_fit(job):

                self.mem_util += job.config['mem_interm_req'] 
                self.cpu_util += job.config['job_util']

                job.start_ts = cur_ts 
                job.end_ts = job.start_ts + job.config['cpu_comp_lat']
                self.jobs += [job]
                return True

            return False

       elif scheme == 'sp-dy':

            if self.does_job_model_fit(job):

                self.mem_util += job.config['mem_interm_req'] + job.config['mem_req']
                self.cpu_util += job.config['job_util']
                job.start_ts = cur_ts

                # TODO: might need to log load latency
                job.end_ts = job.start_ts + job.config['cpu_comp_lat'] + job.config['load_lat']

                self.jobs += [job]
                self.models += [Model(job.stream_id, job.config, cur_ts)]

                return True

            return False

        #elif scheme == 'sp-pre':
            # check if model exists
             

    def rm_job(self, cur_ts):   
 
        finished_jobs = []
        # holds the index of the jobs that needs to be deleted
        finished_pids = []
        for idx, job in enumerate(self.jobs): 

            if job.end_ts < cur_ts:

                finished_jobs += [job]

                # release memory and decrease cpu util 
                self.mem_util -= job.config['mem_interm_req'] 
                self.cpu_util -= job.config['job_util']
                finished_pids += [idx]

        for idx in reversed(finished_pids):
            del self.jobs[idx] # remove process

        return finished_jobs

    def rm_model(self, removed_model, model_id):

        self.mem_util -= removed_model['mem_req']
        del self.models[idx]

    def rm_job_n_model(self, cur_ts):

        finished_jobs = []
        # holds the index of the jobs that needs to be deleted
        finished_pids = []
        for idx, job in enumerate(self.jobs): 

            if job.end_ts < cur_ts:

                finished_jobs += [job]

                # release memory and decrease cpu util 
                self.mem_util -= job.config['mem_interm_req'] 
                self.cpu_util -= job.config['job_util']
                
                # find the model that belongs to the job and rm it 
                for model_id, model in enumerate(self.model):
                    if model.stream_id = job.stream_id and model.create_ts == job.start_ts:
                        rm_model(model, model_id) 
 
                finished_pids += [idx]

        for idx in reversed(finished_pids):
            del self.jobs[idx] # remove jobs

        return finished_jobs

 
class Global:

    def __init__(self, scheme):   
        self.scheme = scheme
        ### 
        param = model_pb2.ApplicationModel()
        with open(SP_MODEL_CONFIG) as f:
            google.protobuf.text_format.Merge(f.read(), param)
            self.model_config = param

        with open(START_FID_PATH) as fh:
            self.sim_start_fids = pickle.load(fh)

        with open(SP_TRACE_LOG) as fh:
            self.sp_trace = pickle.load(fh)

        if SP_CDF_N == 10 and SP_CDF_P == 0.8:
            N_PRE_MACHINES = N_STREAMS * 0.024 
        elif SP_CDF_N == 10 and SP_CDF_P == 0.9:
            N_PRE_MACHINES = N_STREAMS * 0.068 
        ### 

    def get_job_config(self, stream_name, cur_fid):

        if self.scheme == 'nsp':
            compute_flop = float(self.model_config.models[0].compute)
            mem_req = float(self.model_config.models[0].memory)
            load_lat = self.model_config.models[0].loading_latency
            gpu_comp_lat = self.model_config.models[0].compute_latency
            cpu_comp_lat = self.model_config.models[0].cpu_compute_latency
            s_comp_lat = self.model_config.models[0].s_compute_latency
            s_load_lat = self.model_config.models[0].s_loading_latency
            
     
            job_config = {'compute_flop': compute_flop, 'mem_interm_req': INTERM_MEM_REQ, 'mem_req': mem_req, 'load_lat': load_lat, 'gpu_comp_lat': gpu_comp_lat, 'cpu_comp_lat': cpu_comp_lat, 's_comp_lat': s_comp_lat, 's_load_lat': s_load_lat} 

            return job_config

    def get_general_model_config(self):

        compute_flop = float(self.model_config.models[0].compute)
        mem_req = float(self.model_config.models[0].memory)
        load_lat = self.model_config.models[0].loading_latency
        gpu_comp_lat = self.model_config.models[0].compute_latency
        cpu_comp_lat = self.model_config.models[0].cpu_compute_latency
        s_comp_lat = self.model_config.models[0].s_compute_latency
        s_load_lat = self.model_config.models[0].s_loading_latency
 
        model_config = {'compute_flop': compute_flop, 'mem_req': mem_req, 'load_lat': load_lat, 'gpu_comp_lat': gpu_comp_lat, 'cpu_comp_lat': cpu_comp_lat, 's_comp_lat': s_comp_lat, 's_load_lat': s_load_lat} 
        return model_config

    def get_sp_model_config(self, model_type = 7):

        compute_flop = float(self.model_config.models[model_type].compute)
        mem_req = float(self.model_config.models[model_type].memory)
        load_lat = self.model_config.models[model_type].loading_latency
        gpu_comp_lat = self.model_config.models[model_type].compute_latency
        cpu_comp_lat = self.model_config.models[model_type].cpu_compute_latency
        s_comp_lat = self.model_config.models[model_type].s_compute_latency
        s_load_lat = self.model_config.models[model_type].s_loading_latency
 
        model_config = {'compute_flop': compute_flop, 'mem_req': mem_req, 'load_lat': load_lat, 'gpu_comp_lat': gpu_comp_lat, 'cpu_comp_lat': cpu_comp_lat, 's_comp_lat': s_comp_lat, 's_load_lat': s_load_lat} 
            
class Cloud:

    def __init__(self, scheme): 
        self.n_machine = N_MACHINE
        self.scheme = scheme
        self.fairness_count = [0] * N_STREAMS 
 
        self.machines = {}
        for i in xrange(self.n_machine):
            self.machines[i] = Machine(i)

         
    def preload(self):

        if self.scheme == 'nsp': 
            # preload general dnn on all machines
            model_config = GLOBAL.get_general_model_config()
            for mid in self.machines:
                machine = self.machines[mid]
                status = machine.preload_model(-1, model_config)
                if not status:
                    logger.error('Error preloading Model(%s) to Machine(%s)', model_config, machine)
                    exit(-1)

        elif self.scheme == 'sp-dy' or self.scheme == 'sp-pre':
            # compute the portion of non-specialized
            # preload general dnn
            model_config = GLOBAL.get_general_model_config()
            for i, mid in enumerate(self.machines):
                if i == N_PRE_MACHINES:
                    break
                machine = self.machines[mid]
                status = machine.preload_model(-1, model_config)
                if not status:
                    logger.error('Error preloading Model(%s) to Machine(%s)', model_config, machine)
                    exit(-1)

 
    def get_machine_statues(self):

        machine_statuses = {}
        for mid in self.machines:
            machine = self.machines[mid]
            machine_statuses[mid] = machine.get_status()

        return machine_statuses

    def get_cloud_util(self):

        machine_statuses = self.get_machine_statues()
        total_cpu_util, total_mem_util, total_cpu_aval, total_mem_aval = [0.0] * 4
        for mid in machine_statuses:          
            total_mem_util += machine_statuses[mid]['mem_util_byte']
            total_mem_aval += machine_statuses[mid]['mem_lim']
            
            total_cpu_util += machine_statuses[mid]['cpu_util_flop']
            total_cpu_aval += machine_statuses[mid]['cpu_lim']

        cloud_util = {'mem_util_prec': total_mem_util/(total_mem_aval * 1.0), 'cpu_util_prec': total_cpu_util/(total_cpu_aval * 1.0)}

        return cloud_util

    def does_job_model_fit(self, machine_status, job):

        if machine_status['mem_util_byte'] + job.config['mem_interm_req'] + job.config['mem_req'] > machine_status['mem_lim']
            return False
 
        if machine_status['cpu_util_flop'] + job.config['job_util']  > machine_status['cpu_lim']:
            #logger.debug('Exceed cpu util -- CPU util: %f, job util: %f', machine_status['cpu_util_flop'],  job.config['job_util'])
            return False

        return True

    def does_job_fit(self, machine_status, job):
       
         
        if machine_status['mem_util_byte'] + job.config['mem_interm_req'] > machine_status['mem_lim']:
            #logger.debug('Exceed mem util -- mem util: %f, job mem req: %f', machine_status['mem_util_byte'] , job.config['mem_req'])
            return False

        if machine_status['cpu_util_flop'] + job.config['job_util']  > machine_status['cpu_lim']:
            #logger.debug('Exceed cpu util -- CPU util: %f, job util: %f', machine_status['cpu_util_flop'],  job.config['job_util'])
            return False

        return True

    def SLOTBASED_FAIRNESS(self, job_queue):

        fair_share = sum(self.fairness_count)/(len(self.fairness_count) * 1.0)
        dist_from_fair_share = [(fair_share - x/(len(self.fairness_count) * 1.0)) for x in self.fairness_count] # bigger the poorer 
        fair_pair = sorted((fv,i) for i, fv in enumerate(dist_from_fair_share))  # small to big
        fair_rank_index = list(reversed([x[1] for x in fair_pair]))

        # reorder job queue
        tmp_job_queue = list(job_queue)
        ordered_job_queue = []
        for idx in fair_rank_index: 
            all_matchine_jobs = [] 
            for jid, job in enumerate(tmp_job_queue):
                if job.stream_id == idx:
                    all_matchine_jobs += [(jid, job)]
            
            if len(all_matchine_jobs) > 0:
                # multiple matched jobs for a stream, the one emitted first gets processed  
                selected_job = sorted(all_matchine_jobs, key = lambda x: x[1].emit_ts)[0]
                ordered_job_queue += [selected_job[1]]
                del tmp_job_queue[selected_job[0]]

        for job in tmp_job_queue:
            ordered_job_queue += [job]
 
        return ordered_job_queue

    def SRTF(self, job_queue, cur_ts):

        if self.scheme == 'nsp':

            remaining_time = [(cur_ts + j.config['cpu_comp_lat']) - j.emit_ts for j in job_queue]  
            remaining_time_pair = sorted((rt,i) for i, rt in enumerate(remaining_time))  # small to big
            remaining_rank_index = [x[1] for x in remaining_time_pair]
            
            tmp_job_queue = list(job_queue)
            for i, ridx in enumerate(remaining_rank_index):
                job_queue[i] = tmp_job_queue[ridx]

            return job_queue
            

        elif self.scheme == 'sp-dy':
            remaining_time = [(cur_ts + j.config['load_lat'] + j.config['cpu_comp_lat']) - j.emit_ts for j in job_queue]  
            remaining_time_pair = sorted((rt,i) for i, rt in enumerate(remaining_time))  # small to big
            remaining_rank_index = [x[1] for x in remaining_time_pair]
            
            tmp_job_queue = list(job_queue)
            for i, ridx in enumerate(remaining_rank_index):
                job_queue[i] = tmp_job_queue[ridx]

            return job_queue

    def schedule(self, job_queue, cur_ts):

        if len(job_queue) == 0:
            return job_queue

        if self.scheme == 'nsp':

            n_jobs = len(job_queue)

            if SCHEDULE_METRIC == 'completion': 
                job_queue = self.SRTF(job_queue, cur_ts)

            for ite in xrange(n_jobs):
 
                if SCHEDULE_METRIC == 'fairness': 
                    job_queue = self.SLOTBASED_FAIRNESS(job_queue)
                
                job = job_queue[0] 
                machine_statuses = self.get_machine_statues()

                for mid in machine_statuses:
                    if self.does_job_fit(machine_statuses[mid], job):
                        status = self.machines[mid].run(self.scheme, job, cur_ts)
                        if not status:
                            logger.error('Error placing Job(%s) on Mach(%s)', job, self.machines[mid] )
                            exit(-1)
                        else: # job start running
                            self.fairness_count[job.stream_id] += 1
                            ###
                            if len(start_jobs) == 0:
                                logger.info('%d -- Schedule', cur_ts)                            
                            ###

                            logger.info('%d -- Schedule Job(%s) on Mach(%s)', cur_ts, job, self.machines[mid])
                            del job_queue[0] 
                            break

            return job_queue

        elif self.scheme == 'sp-dy':

            n_jobs = len(job_queue)

            if SCHEDULE_METRIC == 'completion':
                job_queue = self.SRTF(job_queue, cur_ts)
            
            for ite in xrange(n_jobs):
 
                if SCHEDULE_METRIC == 'fairness': 
                    job_queue = SLOTBASED_FAIRNESS(job_queue)
                
                job = job_queue[0] 
                machine_statuses = self.get_machine_statues()

                for mid in machine_statuses:
                    if self.does_job_model_fit(machine_statuses[mid], job):
                        status = self.machines[mid].run(self.scheme, job, cur_ts)
                        if not status:
                            logger.error('Error placing Job(%s) on Mach(%s)', job, self.machines[mid] )
                            exit(-1)
                        else: # job start running
                            self.fairness_count[job.stream_id] += 1
                            ###
                            if len(start_jobs) == 0:
                                logger.info('%d -- Schedule', cur_ts)                            
                            ###

                            logger.info('%d -- Schedule Job(%s) on Mach(%s)', cur_ts, job, self.machines[mid])
                            del job_queue[0] 
                            break

            return job_queue
                

    def update(self, cur_ts):


        if self.scheme == 'nsp' or self.scheme == 'sp-pre':
            finished_jobs = []
            for m in self.machines:
                fjs = self.machines[m].rm_job(cur_ts)
                
                ### For debugging ###
                if len(finished_jobs) == 0 and len(fjs) > 0:
                    logger.info('%d -- Update', cur_ts)
                 
                if len(fjs) > 0: 
                    fj_str = ''
                    for fj in fjs:
                        fj_str += 'Job(' + str(fj) + ')\n'
                 
                    logger.info('Mach(%s) finished processing %s', self.machines[m], fj_str)
                ###################
                finished_jobs += fjs
 
            return finished_jobs

        elif self.scheme == 'sp-dy':

            finished_jobs = [] 
            for m in self.machines:
                fjs = self.machines[m].rm_job_n_model(cur_ts)

                ### For debugging ###
                if len(finished_jobs) == 0 and len(fjs) > 0:
                    logger.info('%d -- Update', cur_ts)
                 
                if len(fjs) > 0: 
                    fj_str = ''
                    for fj in fjs:
                        fj_str += 'Job(' + str(fj) + ')\n'
                 
                    logger.info('Mach(%s) finished processing %s', self.machines[m], fj_str)
                ###################
                finished_jobs += fjs
 
            return finished_jobs

 
class Global:

    def __init__(self, scheme):   
        self.scheme = scheme
        ### 
        model_config = ConfigParser.ConfigParser()
        model_config.read('sim.conf')
        models_as_rsc_file = model_config.get('specialize', 'resource_file')
 
        param = model_pb2.ApplicationModel()
        with open(models_as_rsc_file) as f:
            google.protobuf.text_format.Merge(f.read(), param)
        
        self.model_config = param

        with open(DICTIONARY_PATH) as fh:
            self.all_objs = pickle.load(fh)

        with open(START_FID_PATH) as fh:
            self.sim_start_fids = pickle.load(fh)
        ###

    def get_job_config(self, stream_name, cur_fid):

        if self.scheme == 'nsp':
            compute_flop = float(self.model_config.models[0].compute)
            mem_req = float(self.model_config.models[0].memory)
            load_lat = self.model_config.models[0].loading_latency
            gpu_comp_lat = self.model_config.models[0].compute_latency
            cpu_comp_lat = self.model_config.models[0].cpu_compute_latency
            s_comp_lat = self.model_config.models[0].s_compute_latency
            s_load_lat = self.model_config.models[0].s_loading_latency
            
     
            job_config = {'compute_flop': compute_flop, 'mem_interm_req': INTERM_MEM_REQ, 'mem_req': mem_req, 'load_lat': load_lat, 'gpu_comp_lat': gpu_comp_lat, 'cpu_comp_lat': cpu_comp_lat, 's_comp_lat': s_comp_lat, 's_load_lat': s_load_lat} 

            return job_config

    def get_model_config(self):

        if self.scheme == 'nsp':
            compute_flop = float(self.model_config.models[0].compute)
            mem_req = float(self.model_config.models[0].memory)
            load_lat = self.model_config.models[0].loading_latency
            gpu_comp_lat = self.model_config.models[0].compute_latency
            cpu_comp_lat = self.model_config.models[0].cpu_compute_latency
            s_comp_lat = self.model_config.models[0].s_compute_latency
            s_load_lat = self.model_config.models[0].s_loading_latency
     
            job_config = {'compute_flop': compute_flop, 'mem_req': mem_req, 'load_lat': load_lat, 'gpu_comp_lat': gpu_comp_lat, 'cpu_comp_lat': cpu_comp_lat, 's_comp_lat': s_comp_lat, 's_load_lat': s_load_lat} 
            return job_config


def run_scheme(scheme):

    #####
    # load traces
    videos = open(VIDEO_LIST).read().split() 
    stream_trace = {}
    stream_nframes = {}
    for video_name in videos:
        
        stream_nframes[video_name] = get_video_frame_num(video_name)
        
        # load greedy subsampled frames
        greedy_gt_path = os.path.join(GREEDY_TRACE, video_name +  '_' + str(GREEDY_THRESH) + '_gtframe.pickle')
        with open(greedy_gt_path) as gt_fh:
            selected_frame_obj = pickle.load(gt_fh)
            stream_trace[video_name] = selected_frame_obj['picked_f']

    # select N_STREAMS
    streams = []
    streams += videos
    while True:
        if len(streams) > N_STREAMS:
            break
        streams += videos

    streams = random.sample(streams, N_STREAMS)
    #####

    #####
    cloud = Cloud(scheme)
    cloud.preload()
    #####     

    # start streaming
    cur_ts = 0
    pid = 0
    
    job_queue = []
    stream_process_log = dict([(key, {}) for key in range(len(streams))])
    cloud_util_log = {} # key: ts, value: cloud_util
    print 'Simulation starts'

    '''
    for stream_name in streams:
        print stream_name, stream_trace[stream_name]
    '''
    while (cur_ts < SIM_LEN):
        delta_fid = cur_ts / 33
        
        # add jobs into queue
        has_added_job = False
        for sid, stream_name in enumerate(streams):

            # randomize stream start time
            cur_fid = (GLOBAL.sim_start_fids[stream_name] + delta_fid) % stream_nframes[stream_name]

            if cur_ts % 33 == 0 and cur_fid in stream_trace[stream_name]:

                # create a job
                job_config = GLOBAL.get_job_config(stream_name, cur_fid)
                job = Job(pid, sid, cur_fid, cur_ts, -1, -1, job_config)

                if cur_ts in stream_process_log[sid].keys():
                    logger.error('This job has been dispatched before')
                    eixt(-1)
                stream_process_log[sid][cur_ts] = (-1, -1)

                pid += 1
                ###### debug #######
                job_queue += [job]
                has_added_job = True
                ####################

        ###### debug ######
        if has_added_job:
            job_str = ''
            for job in job_queue:
                job_str += str(job) + '\n'
            logger.info('%d -- job queue: %s', cur_ts, job_str)
        ####################

        # cloud updates machine occupancy
        finished_jobs = cloud.update(cur_ts)

        for fj in finished_jobs:
            #print fj
            stream_process_log[fj.stream_id][fj.emit_ts] = (fj.start_ts, fj.end_ts)


        prev_q_size = len(job_queue)
        
        # cloud schedules job in the queue
        job_queue = cloud.schedule(job_queue, cur_ts)
        cloud_util = cloud.get_cloud_util()
        cloud_util_log[cur_ts] = cloud_util
 
        cur_ts += 1 

    print stream_process_log
    plot_cloud_util_log(cloud_util_log) 
    comp_time = compute_ave_job_completion_time(stream_process_log)
    print 'Ave completion time:', np.mean(comp_time), '+-', np.std(comp_time, ddof = 1)


def compute_ave_job_completion_time(stream_process_log):

    comp_time = []
    for stream_id in stream_process_log:
        for emit_ts in stream_process_log[stream_id].keys():
            if stream_process_log[stream_id][emit_ts][1] != -1:
                comp_time += [stream_process_log[stream_id][emit_ts][1] - emit_ts]

    return comp_time

def plot_cloud_util_log(cloud_util_log):
    
    ts = sorted(cloud_util_log.keys())
    fig = plt.figure() 
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ts, [cloud_util_log[x]['mem_util_prec'] * 100 for x in cloud_util_log], color = 'b', label = 'Memory') 
    ax.plot(ts, [cloud_util_log[x]['cpu_util_prec'] * 100 for x in cloud_util_log], color = 'r', label = 'CPU')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Utilization (%)') 
    ax.legend(loc = 'best')
    plt.show()

if __name__ == "__main__":

    scheme = 'nsp'
    global GLOBAL
    GLOBAL = Global(scheme)
    run_scheme(scheme) 
                  
           
