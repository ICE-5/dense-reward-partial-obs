import pyaraas as pa
from pyaraas import Transform, Task
import asyncio

async def pick(workcell):
	cam0 = workcell.get_camera("SR300-0")
	gripper0 = workcell.get_gripper("RqThin-0")
	gripper1 = workcell.get_gripper("RqThin-1")

	# pile some parts at the origin
	await workcell.pile_parts("pile_of_blocks", 0,0,0, [{'name':'BlockA', 'num':10},{'name':'BlockB', 'num':5}])
	
	# try and pick up all the parts (starting with BlockA type)
	fail_cnt = 0
	part_name = "BlockA"
	while True:
		status = await gripper0.grasp(part_name, "pile-yinan4", cam0, Transform(0,0,250,3.14159,0.00,0.00))
		if status == "finished":
			fail_cnt = 0
			await gripper0.move_to(Transform(-2.15,-313.42,321.56,0,0,-3.14159), 1)
			await gripper0.drop_part()
		else:
			if fail_cnt > 0:
				return
			fail_cnt = 1	
			# on the first failure, switch to the other type of part
			if part_name == "BlockA":
				part_name = "BlockB"
			else:
				part_name = "BlockA"

try:
	workcell = pa.start("BrickBot")

	pa.run(Task(pick,workcell))

except Exception as error:
	print("SCRIPT ERROR: " + str(error))
finally:
	pa.stop()