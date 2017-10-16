import matplotlib.pyplot as plt


k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 29, 35, 39, 45, 55]
acc = [47.37, 51.31, 42.10, 57.23, 53.95, 49.34, 46.71, 52.63, 47.37, 50.66, 54.61, 46.71, 48.68, 45.39, 46.37, 50.66]
plt.scatter(k, acc, label='Eval complexity', s=100)
plt.title('Accuracy with respect to number of neighbours')
plt.xlabel('# neighbours')
plt.ylabel('Classification accuracy (%)')
plt.show()