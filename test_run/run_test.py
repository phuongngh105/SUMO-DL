import traci

traci.start(["sumo", "-c", "test.sumocfg"])

for step in range(600):
    traci.simulationStep()
    if step % 100 == 0:
        n = traci.vehicle.getIDCount()
        print(f"Bước {step}: có {n} xe trong mạng")

traci.close()
print("Chạy xong")