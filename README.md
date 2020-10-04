# minimum_surface_triangle
  Given a set of 2D random points and considering one of them as central point, randomly as well, find the triangle with the smallest surface. It is basically a Knn implementation with a "non-linear" restriction. Utilizing the whole shared memory of the Geforce 820Mm, for read-n-write acceleration, and global memmory if shared is about to overflow. Comparison between CPU and GPU implementation, maximum number of points: 131072.   
