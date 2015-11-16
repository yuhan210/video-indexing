import model_pb2
import google.protobuf.text_format
import ConfigParser
import logging
import pickle
import random
import os


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

config = ConfigParser.ConfigParser()
config.read('sim.conf')

N_STREAMS = int(config.get('sim', 'n_streams'))
SIM_LEN = int(config.get('sim', 'sim_len'))
MACHINE_MEMORY = float(config.get('machine', 'memory'))
MACHINE_CORE = int(config.get('machine', 'n_core'))
MACHINE_COMPUTE = float(config.get('machine', 'flop'))
N_MACHINE = int(config.get('machine', 'n_machine'))
GREEDY_TRACE = config.get('trace', 'greedy_trace')
GREEDY_THRESH = float(config.get('trace', 'thresh'))
VIDEO_LIST = config.get('trace', 'video_list')



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

#TODO: log machine utilizations
class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        
        self.MEM_LIM = MACHINE_MEMORY
        self.CPU_LIM = MACHINE_COMPUTE

        self.mem_util = 0 
        self.core = MACHINE_CORE
        self.cpu_util = 0
   
        self.jobs = [] 
        self.cpu_util_log = []  
 
    def __str__(self):
        return 'machine_id:' + str(self.machine_id) + ' mem_util (%):' + str((self.mem_util* 100)/self.MEM_LIM) + ' cpu_util (%):' + str((self.cpu_util * 100)/self.CPU_LIM)

    def get_status(self):

        m_status = {'mem_util_byte': self.mem_util, 'mem_util_prec': self.mem_util/(self.MEM_LIM * 1.0), 'cpu_util_flop': self.cpu_util, 'cpu_util_prec': self.cpu_util/(self.CPU_LIM * 1.0), 'mem_lim': self.MEM_LIM, 'cpu_lim': self.CPU_LIM}

        return m_status

    def does_job_fit(self, job):
       
        # TODO: one process per core? 
        # check if adding this job exceeds mem capacity
        if self.mem_util + job.config['mem_req'] > self.MEM_LIM :
            return False

        # check if adding this job exceeds cpu util
        if self.cpu_util + job.config['job_util'] > self.CPU_LIM:
            return False

        return True 

    def run(self, scheme, job, cur_ts):

        if scheme == 'nsp': 
            if self.does_job_fit(job):
                self.mem_util += job.config['mem_req'] 
                # TODO: verify the variable
                self.cpu_util += job.config['job_util']
                job.start_ts = cur_ts 
                job.end_ts = job.start_ts + job.config['cpu_comp_lat']
                self.jobs += [job]
                return True
            return False

    def update(self, cur_ts):   
 
        finished_jobs = []
        # holds the index of the jobs that needs to be deleted
        finished_pids = []
        for idx, job in enumerate(self.jobs): 
            if job.end_ts < cur_ts:

                finished_jobs += [job]

                # release memory and decrease cpu util 
                self.mem_util -= job.config['mem_req'] 
                self.cpu_util -= job.config['job_util']
                finished_pids += [idx]

        for idx in reversed(finished_pids):
            del self.jobs[idx] # remove process

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
        ###

    def get_job_config(self):

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

class Cloud:

    def __init__(self, scheme): 
        self.n_machine = N_MACHINE
        self.scheme = scheme
       
        self.machines = {}
        for i in xrange(self.n_machine):
            self.machines[i] = Machine(i)


    def get_machine_statues(self):

        machine_statuses = {}
        for mid in self.machines:
            machine = self.machines[mid]
            machine_statuses[mid] = machine.get_status()

        return machine_statuses

    def get_cloud_util(self):

        machine_statuses = self.get_machine_statues()
        for mid in machine_statuses:          
            machine_statuses[mid] 

    def does_machine_fit(self, machine_status, job):
       
         
        if machine_status['mem_util_byte'] + job.config['mem_req'] > machine_status['mem_lim']:
            #logger.debug('Exceed mem util -- mem util: %f, job mem req: %f', machine_status['mem_util_byte'] , job.config['mem_req'])
            return False

        if machine_status['cpu_util_flop'] + job.config['job_util']  > machine_status['cpu_lim']:
            #logger.debug('Exceed cpu util -- CPU util: %f, job util: %f', machine_status['cpu_util_flop'],  job.config['job_util'])
            return False

        return True

    def schedule(self, job_queue, cur_ts):
        if self.scheme == 'nsp':
            # FCFS (cause all jobs have the same remaining time) 
            job_queue = sorted(job_queue, key = lambda x: x.start_ts)

            start_jobs = [] 
            for jid, job in enumerate(job_queue):

                machine_statuses = self.get_machine_statues()

                for mid in machine_statuses:
                    if self.does_machine_fit(machine_statuses[mid], job):
                        success = self.machines[mid].run(self.scheme, job, cur_ts)
                        if not success:
                            logger.error('Error placing Job(%s) on Mach(%s)', job, self.machines[mid] )
                            exit(-1)
                        else:
                            # schedule the next job
                            ###
                            if len(start_jobs) == 0:
                                logger.info('%d -- Schedule', cur_ts)                            
                            ###

                            logger.info('%d -- Schedule Job(%s) on Mach(%s)', cur_ts, job, self.machines[mid])
                            start_jobs += [jid]    
                            break
            for jid in reversed(start_jobs):
                del job_queue[jid]

            return job_queue

    def update(self, cur_ts):
        if self.scheme == 'nsp':
            finished_jobs = []
            for m in self.machines:
                fjs = self.machines[m].update(cur_ts)
                
                ###
                if len(finished_jobs) == 0 and len(fjs) > 0:
                    logger.info('%d -- Update', cur_ts)
             
                if len(fjs) > 0: 
                    fj_str = ''
                    for fj in fjs:
                        fj_str += 'Job(' + str(fj) + ')\n'
                 
                    logger.info('Mach(%s) finished processing %s', self.machines[m], fj_str)
                ###
                finished_jobs += fjs
 
            return finished_jobs


        

def run_scheme(scheme):

    #####
    # load traces
    videos = open(VIDEO_LIST).read().split() 
    stream_trace = {}
    #stream_nframes = {}
    for video_name in videos:
        
        #stream_nframes[video_name] = get_video_frame_num(video_name)
        
        # load greedy subsampled frames
        greedy_gt_path = os.path.join(GREEDY_TRACE, video_name +  '_' + str(GREEDY_THRESH) + '_gtframe.pickle')
        with open(greedy_gt_path) as gt_fh:
            selected_frame_obj = pickle.load(gt_fh)
            stream_trace[video_name] = selected_frame_obj['picked_f']

    # select N_STREAMS (TODO: might need to determine whether this stream is specializable)i
    streams = []
    streams += videos
    while True:
        if len(streams) > N_STREAMS:
            break
        streams += videos

    streams = random.sample(streams, N_STREAMS)
    #####

    cloud = Cloud(scheme)
    
    # start streaming
    cur_ts = 0
    pid = 0
    
    job_queue = []
    stream_process_log = dict([(key, {}) for key in range(len(streams))])
    #cloud_util_log  
    #stream_process_log = dict.fromkeys(range(len(streams))) # a dict (key: sid) of dict (key: start_ts, value: end_ts)
    print 'Simulation starts'

    '''
    for stream_name in streams:
        print stream_name, stream_trace[stream_name]
    '''
    while (cur_ts < SIM_LEN):
        cur_fid = cur_ts / 33
        
        # add jobs into queue
        has_added_job = False
        for sid, stream_name in enumerate(streams):
            if cur_ts % 33 == 0 and cur_fid in stream_trace[stream_name]:

                # create a job
                job_config = GLOBAL.get_job_config()
                job = Job(pid, sid, cur_fid, cur_ts, -1, -1, job_config)
                if cur_ts in stream_process_log[sid].keys():
                    logger.error('This job has been dispatched before')
                    eixt(-1)
                stream_process_log[sid][cur_ts] = (-1, -1)

                pid += 1
                job_queue += [job]
                has_added_job = True

        if has_added_job:
            job_str = ''
            for job in job_queue:
                job_str += str(job) + '\n'
            logger.info('%d -- job queue: %s', cur_ts, job_str)
            
        # cloud updates machine occupancy
        finished_jobs = cloud.update(cur_ts)

        for fj in finished_jobs:
            #print fj
            stream_process_log[fj.stream_id][fj.emit_ts] = (fj.start_ts, fj.end_ts)


        prev_q_size = len(job_queue)
        
        # cloud schedules job in the queue
        job_queue = cloud.schedule(job_queue, cur_ts)

        machine_statuses = cloud.get_machine_statues() 
        cur_ts += 1 

    print stream_process_log
        
if __name__ == "__main__":

    scheme = 'nsp'
    global GLOBAL
    GLOBAL = Global(scheme)
    run_scheme(scheme) 
                  
           
