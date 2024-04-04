import os
import operator


# Define the folder sizes and their corresponding names
lang2size = {}
with open("./folder_size.txt", 'r') as fp:
    for line in fp:
        if line.strip() == "":
            continue
        size, lang = line.split()
        lang2size[lang] = int(size)


# Sort the folders by size in descending order
sorted_folders = sorted(lang2size.items(), key=operator.itemgetter(1), reverse=True)

# Initialize five categories
categories = [[] for _ in range(6)]

# Distribute the folders into categories
for folder, size in sorted_folders:
    # Find the category with the smallest sum of sizes
    min_category = min(categories, key=lambda category: sum(lang2size.get(f, 0) for f in category))
    min_category.append(folder)

# Print the result
for i, category in enumerate(categories):
    total_size = sum(lang2size.get(f, 0) for f in category)
    print(f"Category {i + 1}: {category}, Total Size: {total_size:.1f}G")