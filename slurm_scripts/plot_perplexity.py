import numpy as np
import matplotlib.pyplot as plt

# Cross-entropy loss values from the log
loss_values = [
    5.4226, 5.1251, 4.9748, 4.8961, 4.8427,
    4.7956, 4.7634, 4.7353, 4.7068, 4.6891
]

# Calculate perplexity for each loss value
perplexity_values = np.exp(loss_values)

# Plotting the graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(perplexity_values) + 1), perplexity_values, marker='o', color='b', label='Perplexity')
plt.title('Perplexity over Evaluation Metrics')
plt.xlabel('Evaluation Step')
plt.ylabel('Perplexity')
plt.xticks(range(1, len(perplexity_values) + 1))
plt.grid(True)
plt.legend()
plt.show()

