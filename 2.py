import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the data
names = {0: 'down', 1: 'lookaround', 2: 'phone', 3: 'up'}
counts = [10, 20, 30, 40, 50, 60]  # Example data, you can modify these numbers based on your actual situation

# Create the bar chart
fig, ax = plt.subplots()
bars = ax.bar(names.keys(), counts, tick_label=names.keys())


# Initialize function
def init():
    for bar in bars:
        bar.set_height(0)
    return bars


# Animation update function
def update(frame):
    # Calculate the percentages
    total = sum(counts)
    percentages = [(count / total) * 100 for count in counts]
    # Update the bar heights
    for bar in bars:
        bar.set_height(percentages[bar.get_x()] - percentages[bar.get_x() - ax.spines["left"].get_bounds()[0]])
    return bars


# Create the animation object
ani = animation.FuncAnimation(fig, update, frames=range(1, len(counts) + 1), init_func=init, blit=True)

# Set the title and labels
plt.title('Six categories situation statistics (percentage)')
plt.xlabel('Category')
plt.ylabel('Percentage')

# Show the plot
plt.show()
