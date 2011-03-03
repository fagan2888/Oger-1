import mdp.parallel
import pp
import subprocess
import getpass
from datetime import datetime

class GridScheduler(mdp.parallel.Scheduler):
    def __init__(self, ppserver=None, max_queue_length=1,
                 result_container=mdp.parallel.ListResultContainer(), ppservers=None,
                 verbose=False):
        super(GridScheduler, self).__init__(result_container=result_container,
                                          verbose=verbose)
        self.ppserver = ppserver
        self.jobs = []
        self.job_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        try:
            self.whoami = getpass.getuser()
        except:
            print 'Unable to determine username'
            raise

        # List of tuples instead of a dictionary because the order of the commands
        # is apparently important for condor
        self.condor_options = [
        ['cmd', '/opt/Python-2.6.4/bin/ppserver.py'], \
        ['args', '-p 60002 -t 1 -s '], \
        ['log', '/tmp/log_condor_pp_test.log'], \
        ['requirements', 'OpSys=="*" || Arch=="*" || FileSystemDomain == "elis.ugent.be"'], \
        ['output', '/mnt/snn_gluster/Oger_jobs/dvrstrae/out.$(cluster)'], \
        ['universe', 'vanilla'], \
        ['environment', '"MPLCONFIGDIR=/tmp/"'], \
        ['Notification', 'Error'], \
        ['should_transfer_files', 'No'], \
        #('when_to_transfer_output', 'ON_EXIT'), \
        ['transfer_executable', 'False'], \
        ['InitialDir', '/mnt/snn_gluster/Oger_jobs/' + self.whoami], \
        ]

        if ppservers is None:
            self.ppservers = ('clsnn001:60002',
                 'clsnn002:60002',
                 'clsnn003:60002',
                 'clsnn004:60002',
                 'clsnn005:60002',
                 'clsnn006:60002',
                 'clsnn007:60002',
                 'clsnn008:60002',
                 'clsnn009:60002',
                 'clsnn010:60002',
                 'clsnn011:60002',
                 'clsnn012:60002',
                 'clsnn013:60002',
                 'clsnn014:60002',
                 'clsnn015:60002',
                 'clsnn016:60002',
                 'clsnn017:60002',
                 'clsnn018:60002',
                 'clsnn019:60002',
                 'clsnn020:60002',
                 'clsnn021:60002',
                 'clsnn022:60002',
                 'clsnn023:60002',
                 'clsnn024:60002',
                 'clsnn025:60002',
                 'clsnn026:60002',
                 'clsnn027:60002',
                 'clsnn028:60002',
                 'clsnn029:60002',
                 'clsnn030:60002',
                 )
        else:
            self.ppservers = ppservers

        if ppserver is None:
            self.ppserver = pp.Server(ncpus=0, ppservers=self.ppservers, secret=self.whoami + '_' + self.job_id)
        else:
            self.ppserver = ppserver


        self.jobs_done = 0

    def execute(self):
        tempfile = open('condor_tmp_file.job', 'w')
        self.condor_options[1][1] += self.whoami + '_' + self.job_id
        self.condor_job_str = ''
        for option, value in self.condor_options:
            self.condor_job_str += option + ' = ' + value + '\n'

        self.condor_job_str += 'queue ' + str(len(self.jobs))
        tempfile.write(self.condor_job_str)
        tempfile.close()
        p = subprocess.Popen(['condor_submit', 'condor_tmp_file.job'], stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        print self.condor_job_str
        stdout = p.communicate()[0]
        print(stdout)

    def _process_task(self, data, task_callable, task_index):
        """Non-blocking processing of tasks.

        Depending on the scheduler state this function is non-blocking or
        blocking. One reason for blocking can be a full task-queue.
        """
        task = (data, task_callable.fork(), task_index)
        def execute_task(task):
            """Call the first args entry and return the return value."""
            data, task_callable, task_index = task
            task_callable.setup_environment()
            return task_callable(data), task_index
        task_submitted = False
        while not task_submitted:
            # release lock to enable result storage
            self._lock.release()
            # the inner tuple is a trick to prevent introspection by pp
            # this forces pp to simply pickle the object
            self.jobs.append(self.ppserver.submit(execute_task, args=(task,),
                                 callback=self._pp_result_callback))
            task_submitted = True

    def get_results(self):
        self.execute()
        return super(GridScheduler, self).get_results()

    def _pp_result_callback(self, result):
        """Calback method for pp to unpack the result and the task id.

        This method then calls the normal _store_result method.
        """
        self.jobs_done += 1
        print "Jobs done: " + str(self.jobs_done) + "/" + str(len(self.jobs))
        self._store_result(result[0], result[1])

    def _shutdown(self):
        """Call destroy on the ppserver."""
        self.ppserver.destroy()
