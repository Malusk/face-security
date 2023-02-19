import math
my_list = []
with open("output.txt", "r") as f:
    for line in f:
        # Strip any trailing newline characters from the line
        line = line.rstrip()
        # Split the line into a list using commas as the delimiter
        values = line.split(",")
        # Convert each value to its appropriate data type and append to my_list
        my_list.append([float(v) for v in values])
distance = []
print(len(my_list))
for a in range(len(my_list)):
    for j in range(a+1,len(my_list)):
        for i in range(len(my_list[0])):
            distance.append(my_list[a][i] - my_list[j][i])
        diffsquare = 0
        for j in distance:
            diffsquare += j**2
        print(math.sqrt(diffsquare))
        distance.clear()