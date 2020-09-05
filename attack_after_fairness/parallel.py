import subprocess, time


class Parallel(object):
	def __init__(self, p=1):
		self.p = p
		self.slots = [None for i in range(p)]
		self.queue = []

	def add_cmd(self, cmd):
		if type(cmd) is list:
			self.queue.extend(cmd)
		else:
			self.queue.append(cmd)

	def run(self, sleep=0.1, info=True, shell=False):
		running = 0
		while True:
			for i in range(self.p):
				if self.slots[i] is None:
					if self.queue:
						cmd = self.queue.pop(0)
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
							self.slots[i] = None
							running -= 1
			if running == 0:
				break
			time.sleep(sleep)
