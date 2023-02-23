import math
def main():
	my_list = []
	with open("output.txt", "r") as f:
		for line in f:
			# Strip any trailing newline characters from the line
			line = line.rstrip()
			# Split the line into a list using commas as the delimiter
			values = line.split(",")
			# Convert each value to its appropriate data type and append to my_list
			my_list.append([float(v) for v in values])

	for i in range(len(my_list)):
		for j in range(i+1,len(my_list)):
			print(euclidean_distance(my_list[i],my_list[j]))
		
def euclidean_distance(x, y):
	return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
	
if __name__ == "__main__":
	main()
