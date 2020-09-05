import subprocess, time, multiprocessing, random, os


class Parallel(object):
	def __init__(self, p=1):
		self.max_cores = multiprocessing.cpu_count()
		self.p = int(min(p, self.max_cores))
		self.slots = [None for i in range(p)]
		self.command = [None for i in range(p)]
		self.cores = [None for i in range(p)]
		self.queue = []

	def add_cmd(self, cmd):
		if type(cmd) is list:
			self.queue.extend(cmd)
		else:
			self.queue.append(cmd)

	def run(self, sleep=0.1, info=True, shell=False, assign_proc=True):
		running = 0
		while True:
			for i in range(self.p):
				if self.slots[i] is None:
					if self.queue:
						cmd = self.queue.pop(0)
						self.command[i] = cmd
						print("Running:", cmd)
						if info:
							self.slots[i] = subprocess.Popen(cmd, shell=shell)
						else:
							self.slots[i] = subprocess.Popen(
								cmd,
								stdout=subprocess.PIPE,
								stderr=subprocess.PIPE,
								shell=shell,
							)
						time.sleep(0.01)
						running += 1

						ready = list(set(range(0, self.max_cores)) - set(self.cores))
						index = random.choice(ready)
						self.cores[i] = index
						proc_string = "0x" + "1" + "0" * index
						os.system(f"taskset -p {proc_string} {self.slots[i].pid}")

				else:
					ret = self.slots[i].poll()
					if ret is not None:
						if ret != 0:
							print(f"Error {ret} with command {cmd}")
							output, error = self.slots[i].communicate()
							print(output)
							print(error)
							for j in range(self.p):
								if self.slots[j] is not None:
									self.slots[j].kill()
							return
						else:
							print("Exited:", self.command[i])
							self.command[i] = None
							self.slots[i] = None
							running -= 1
			if running == 0:
				break
			time.sleep(sleep)
