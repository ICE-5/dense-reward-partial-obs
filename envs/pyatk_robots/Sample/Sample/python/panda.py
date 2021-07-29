import pyaraas as pa
from pyaraas import Transform, Task

async def pick(workcell):
	gripper0 = workcell.get_gripper("PdDefault-0")

	# add a post part to the scene at the given location
	workcell.add_part("BlockB", "PartA", Transform(400,300,25.4,0.00,0.00,0.00))


	# move the kuka end effector to the given location in 1 second (linear end effector path)
	await gripper0.move_to(Transform(400,300,100,0.00,0.00,1.5707), 3)
		
	# open the gripper 50 mm in 1 second
	await gripper0.open(50, 1)

	# move the finger to the given location in 1 second (linear finger path)
	await gripper0.move_to(Transform(400,300,40,0.00,0.00,1.5707), 1)

	# close the gripper around the post 
	await gripper0.close()

	# move the finger along the specified path with waypoints
	path = [{"xform": Transform(400,300,120,0.00,0.00,1.5707), "tick": 2},
			{"xform": Transform(400,530,510,1.5707,0,0), "tick": 4}]
	await gripper0.set_trajectory(path)

	await gripper0.drop_part()

try:
	workcell = pa.start("Panda")

	pa.run(Task(pick,workcell))

except Exception as error:
	print("SCRIPT ERROR: " + str(error))
finally:
	pa.stop()

