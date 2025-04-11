## Assignment 2

# Question 1

The scoring plot identifies three clusters of similar observations, spread
along the axis of PCA 1, which might indicate groups of vehicles that share
underlying characteristics (e.g., similar engine configurations or
performance profiles).

The plot could also potentially highlight prominent outliers, but in this
case, there are none.

Furthermore, the plot could show important trends and gradients, perhaps
reflecting changes over model years or differences by origin. 

A color analysis of the score plot reveals that the clustering likely is
caused by the separation between cars with different number of cylinders.
The plot clearly shows that the three groups reflect cars with 3-5, 6, and 8
cylinders, respectively.  American cars are present in all three groups,
while European and Asian cars are present only in the first group.  There is
also a clear separation between Asian and American cars in the first group.

# Question 2

The loading plot shows how much each variable contributes to the total
variance in the principal components.  For this particular dataset:

The greatest positive values along PCA 1 axis are in Horsepower, Cylinder,
Displacement and Weight (values around 0.4).  The greatest negative values
along the PCA 1 axis is in Miles_per_gallon (values around -0.4).

One could interpret PCA 1 as a contrast between vehicle size/performance and
fuel efficiency.  In other words, vehicles scoring high on PCA 1 tend to have
larger, more powerful engines and heavier weight, which correlates with
lower driving distance, while vehicles with lower PCA 1 scores tend to be
more fuel efficient.

PCA 2 is dominated by the contributions from Model_year, Acceleration and
Origin.

# Question 3

Some benefits include:

Dimensionality reduction: PCA combines correlated variables into a smaller
number of principal components, making it easier to visualize and analyze
the underlying structure.

Noise reduction: PCA can filter out noise by focusing on the components that
capture the most variance, thereby potentially improving subsequent analyses
or modeling.

Handling multicollinearity: When variables are highly correlated, PCA can
create uncorrelated (orthogonal) principal components, which help in
regression or clustering applications.

Enhanced visualization: Reducing the data to a few principal components
makes it possible to generate scatter plots (score plots) that reveal
clustering, outliers, and relationships which might not be obvious in the
original high-dimensional data.
