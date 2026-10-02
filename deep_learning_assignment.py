# Deep Learning Assignment
# Q1: Manually Perform Convolution
# Q3: Show Flattening

# --------------------------------------------------
# QUESTION 1: MANUAL CONVOLUTION
# --------------------------------------------------

image = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

kernel = [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
]

image_size = 5
kernel_size = 3
output_size = image_size - kernel_size + 1

feature_map = []

for i in range(output_size):
    row = []

    for j in range(output_size):
        total = 0

        for ki in range(kernel_size):
            for kj in range(kernel_size):
                total += image[i + ki][j + kj] * kernel[ki][kj]

        row.append(total)

    feature_map.append(row)

print("===== QUESTION 1: MANUAL CONVOLUTION =====")
print("Input Image:")
for row in image:
    print(row)

print("\nKernel:")
for row in kernel:
    print(row)

print("\nFeature Map:")
for row in feature_map:
    print(row)

# First region calculation
print("\nFirst Region Calculation:")
region = [row[:3] for row in image[:3]]

for row in region:
    print(row)

print("\nCalculation:")
print("(0*-1) + (0*-1) + (0*-1) + "
      "(0*0) + (0*0) + (0*0) + "
      "(1*1) + (1*1) + (1*1)")

print("First feature-map value =", feature_map[0][0])


# --------------------------------------------------
# QUESTION 3: FLATTENING
# --------------------------------------------------

matrix = [
    [6, 4],
    [8, 6]
]

flatten_output = []

for row in matrix:
    for value in row:
        flatten_output.append(value)

print("\n\n===== QUESTION 3: FLATTENING =====")
print("Input Matrix:")
for row in matrix:
    print(row)

print("\nFlatten Output:")
print(flatten_output)

print("\nRole of Flatten Layer:")
print("The flatten layer converts a multi-dimensional feature map")
print("into a one-dimensional vector. This vector is then passed")
print("to a fully connected (dense) layer for classification.")
