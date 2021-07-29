import pyaraas as pa
from pyaraas import Transform, Task
import asyncio

async def regrasp(workcell):
	cam0 = workcell.get_camera("SR300-0")
	gripper0 = workcell.get_gripper("RqThin-0")
	gripper1 = workcell.get_gripper("RqThin-1")

	# pile some parts at the origin
	await workcell.pile_parts("pile_of_blocks", 0,0,0, [{'name':'BlockA', 'num':10},{'name':'BlockB', 'num':5}])
	
	# pick up a BlockA part from the pile
	status = await gripper0.grasp("BlockA", "pile-yinan4", cam0, Transform(0,0,250,3.14159,0.00,0.00))
	if status != "finished":
		print("failed to pick up part")
		return

	# regrasp the part so we are holding it with a specific grasp (left index 0)
	result = await gripper0.regrasp("pose-yinan", gripper1, "BlockA", gripper0, "left", 0)
	if "xform" in result.keys():
		print(result["xform"])
		await gripper0.drop_part()
	else:
		print(result)

try:
	workcell = pa.start("BrickBot")

	pa.run(Task(regrasp,workcell))

except Exception as error:
	print("SCRIPT ERROR: " + str(error))
finally:
	pa.stop()