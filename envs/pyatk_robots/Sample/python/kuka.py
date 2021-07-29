import pyaraas as pa
from pyaraas import Transform, Task
	
async def pick(workcell):
	gripper0 = workcell.get_gripper("SchunkDual-0")
	kuka0 = workcell.get_robot("KUKA-KR60-3-0")

	# add a post part to the scene at the given location
	post0 = workcell.add_part("Post", "Post0", Transform(822.77,918.24,4,0.00,0.00,1.5707))	
	post1 = workcell.add_part("Post", "Post1", Transform(822.77,918.24,4,0.00,0.00,0.00))

	# link the two pieces together 
	post0.link([post1], True)

	# move the kuka end effector to the given location in 1 second (linear end effector path)
	await kuka0.move_to(Transform(819,918.24,400.25,3.14159,0.00,0.00), 3)
		
	# open the gripper 120 mm in 1 second
	await gripper0.open(120, 1)

	# move the finger to the given location in 1 second (linear finger path)
	await gripper0.move_to(Transform(824.00,918.24,28,0.00,0.00,-3.14159), 1)

	# close the gripper around the post 
	await gripper0.close()

	await gripper0.move_to(Transform(824.00,918.24,251.35,-0.00,0.00,3.14159), 1)

	# move the finger along the specified path with waypoints
	path = [{"xform": Transform(823.98,1637.05,898.33,-0.00,-0.00,3.14159), "tick": 2},
			{"xform": Transform(823.98,1924.63,1225.09,90.00,-0.00,3.14159), "tick": 4}]
	await gripper0.set_trajectory(path)

try:
	workcell = pa.start()

	pa.run(Task(pick,workcell))

except Exception as error:
	print("SCRIPT ERROR: " + str(error))
finally:
	pa.stop()

