import numpy as np
from math import ceil
from copy import deepcopy
from PIL import Image, ImageDraw
# import cairocffi as cairo
from skimage.measure import compare_ssim as ssim
from random import randint, choice, random
from numba import njit, jit
import time

# function has to live outside of the class to be jitted
@njit
def mse(a, b):
	out = 0
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			for k in range(a.shape[2]):
				out += (a[i][j][k] - b[i][j][k]) ** 2

	out /= (a.shape[0]*a.shape[1])
	return out

class Organism():
	def __init__(self, generation, c, parent, w, h):
		self.generation = generation
		self.id = c
		self.parent = parent
		self.w = w
		self.h = h
		self.genome = []
		self.array = []
		self.fitness = 0
		self.scaled_fitness = 0
		self.Nx = 0
		self.nr = 0
		self.d = 0

	def deepish_copy_genome(self):
		# homemade deepcopy
		newgenome = []

		for gene in self.genome:
			newpoly = []
			for vertex in gene[0]:
				newpoly.append((vertex + tuple()))
			newcol = gene[1] + tuple()
			newgene = (newpoly, newcol)
			newgenome.append(newgene)

		return newgenome

	def initialize_genome(self, genomesize, numvertex):
		# sets a random genome for an organism
		for i in range(0, genomesize):
			poly = []
			for i in range(0,3):
				xy = (randint(0, self.w),randint(0,self.h))
				poly.append(xy)
			# random rgba tuple 11-09 HARDCODED LOW ALPHA VALUE
			color = (randint(0,255),randint(0,255),randint(0,255),randint(0,255))
			# color = (random(),random(),random(),random())
			gene = (poly, color)
			self.genome.append(gene)

		vertices_remaining = numvertex - (genomesize * 3)
		for i in range(0, vertices_remaining):
			j = randint(0, len(self.genome) - 1)
			xy = (randint(0,self.w),randint(0,self.h))
			self.genome[j][0].append(xy)

		# first 150 vertices have been devided over genomesize polygons, ensuring each polygon is at least a triangle
		# remaining vertices are randomly distributed over the polygons
	def name(self):
		return "{:0>6}".format(self.generation) + "-" + "{:0>3}".format(self.id)

	def genome_to_array(self):
		img = Image.new('RGB', (self.w, self.h), color = (0,0,0))
		drw = ImageDraw.Draw(img, 'RGBA')

		for gene in self.genome:
			polygon = gene[0]
			color = gene[1]

			drw.polygon(polygon, color)

		self.array = np.array(img)

	def genome_to_array_cairo(self):
		surface = cairo.ImageSurface(cairo.FORMAT_RGB24, self.w, self.h)
		ctx = cairo.Context(surface)

		ctx.set_source_rgb(0, 0, 0)
		ctx.paint()

		for gene in self.genome:
			ctx.move_to(gene[0][0][0], gene[0][0][1])
			for xy in gene[0][1:]:
				ctx.line_to(xy[0], xy[1])
			#ctx.close_path()
			color = gene[1]
			col = tuple(cl/255 for cl in color)
			ctx.set_source_rgba(*col)
			ctx.stroke()

		buf = surface.get_data()

		a = np.frombuffer(buf, np.uint8)
		a.shape = (self.h, self.w, 4)
		a = a[:,:,:3]

		self.array = a

	# def calculate_fitness_mse(self, goal):
	# 	# calculates mean square error image difference (lower = more similar)
	# 	mse = np.sum((self.array.astype("float") - goal.astype("float")) ** 2)
	# 	mse /= float(self.array.shape[0] * self.array.shape[1])
	# 	# error = np.square(goal - self.array).mean(axis = None)

	# 	self.fitness = mse

	def calculate_fitness_mse(self, goal):
		# calls jitted MSE function declared above this class definition
		self.fitness = mse(self.array, goal)

	def calculate_fitness_ssim(self, goal):
		# calculates structural similarity index image difference (higher = more similar)
		ssim_index = ssim(self.array, goal, multichannel = True)
		self.fitness = ssim_index

	def scale_fitness(self, minimum, maximum):
		# GIVES A PROBLEM IF MIN = MAX
		self.scaled_fitness = (self.fitness - minimum) / (maximum - minimum)

	def calculate_runners(self, nmax, mmax):
		# see Salhi & Fraga 2011

		# mapping fitness to emphasize good solutions
		self.Nx = 0.5 * (np.tanh(4 * self.scaled_fitness - 2) + 1)

		# calculate number of runners (nr) and the distance in number of mutations (d)
		r = random()
		self.nr = int(ceil(nmax * self.Nx * r))
		self.d = int(ceil(mmax * (1 - self.Nx) * r))

	def random_mutation(self, number):
		# performs number random mutations
		# note: function names of all available mutations have to be hardcoded in list 'options' below
		options = [self.gene_jump, self.move_vertex, self.transfer_vertex, self.change_color]

		for i in range(0, number):
			mutation = choice(options)
			mutation()

	def gene_jump(self):
		# moves a polygon in the genome, changing the drawing order
		i = randint(0, len(self.genome) - 1)
		j = randint(0, len(self.genome) - 1)

		gene = self.genome[i]
		del self.genome[i]
		self.genome.insert(j, gene)
		# TODO how does the insertion change the index of the other polygons in the list

	def move_vertex(self):
		# change a (x, y) coord of a vertex of a random generation
		xy = (randint(0,self.w), randint(0,self.h))
		i = randint(0, len(self.genome) - 1)
		v = randint(0, len(self.genome[i][0]) - 1)
		self.genome[i][0][v] = xy

	def transfer_vertex(self):
		# transfers a vertex from polygon i to polygon y
		# places the vertex on a line of receiving polygon to NOT change its shape right away
		giver = 0
		receiver = 0

		# ensure different indexes and ensure that the giver has > 3 vertices
		while True:
			giver = randint(0, len(self.genome) - 1)
			receiver = randint(0, len(self.genome) - 1)
			if giver != receiver and len(self.genome[giver][0]) > 3:
				break

		# pick a vertex from the giver and delete it
		n = randint(0, len(self.genome[giver][0]) - 1)
		del self.genome[giver][0][n]

		# pick two neighbouring vertices from the receiver and interpolate a new (x,y) coordinate between them
		i = randint(0,len(self.genome[receiver][0]) - 2)
		xy1 = self.genome[receiver][0][i]
		xy2 = self.genome[receiver][0][i + 1]

		# calculate the slope of the line between xy1 and xy2
		slope = (xy1[1] - xy2[1]) / (xy1[0] - xy2[0] + 0.00001)

		# pick a random x between x1 and x2, and calculate correponding y. round.
		if xy1[0] < xy2[0]:
			x = randint(xy1[0], xy2[0])
			dx = x - xy1[0]
			y = int(round(dx * slope)) + xy1[1]
		else:
			x = randint(xy2[0], xy1[0])
			dx = x - xy2[0]
			y = int(round(dx * slope)) + xy2[1]

		xy_new = (x, y)
		self.genome[receiver][0].insert(i + 1, xy_new)

	def change_color(self):
		# changes one channel of the color (or the alpha) of a random polygon
		i = randint(0, len(self.genome) - 1)
		j = randint(0, 4)

		if j == 0:
			color = (randint(0,255), self.genome[i][1][1],self.genome[i][1][2], self.genome[i][1][3])
		elif j == 1:
			color = (self.genome[i][1][0], randint(0,255),self.genome[i][1][2], self.genome[i][1][3])
		elif j == 2:
			color = (self.genome[i][1][0], self.genome[i][1][1], randint(0,255), self.genome[i][1][3])
		else:
			color = self.genome[i][1][0], self.genome[i][1][1], self.genome[i][1][2], randint(0,255)
		new_gene = (self.genome[i][0], color)
		self.genome[i] = new_gene

	def save_img(self, directory):
		filename = directory + "/" + "{:0>6}".format(self.generation) + "-" + "{:0>3}".format(self.id) + "-" + str(self.fitness) + ".png"
		im = Image.fromarray(self.array)
		im.save(filename)

	def save_polygons(self, directory):
		filename = directory + "/" + "{:0>6}".format(self.generation) + "-" + "{:0>3}".format(self.id) + "-" + str(self.fitness) + ".txt"

		with open(filename, 'w') as f:
			for poly in self.genome:
   			 	f.write(str(poly) + '\n')

	def save_img_vectorized(self):
		surface = cairo.SVGSurface('test.svg', self.w, self.h)
		ctx = cairo.Context(surface)

		ctx.set_source_rgb(0, 0, 0)
		ctx.paint()

		for gene in self.genome:
			ctx.move_to(gene[0][0][0], gene[0][0][1])
			for xy in gene[0][1:]:
				ctx.line_to(xy[0], xy[1])
			ctx.close_path()
			color = gene[1]
			col= tuple(cl/255 for cl in color)
			ctx.set_source_rgba(*col)
			ctx.fill()

		surface.finish()

# im_goal = Image.open("starry-night-498-402.jpg")
# goal = np.array(im_goal)


# # start = time.time()

# # for i in range(0, 100):
# # 	alex = Organism(i, 2, "test", 498, 402)
# # 	alex.initialize_genome(50, 400)
# # 	alex.genome_to_array_cairo()

# # end = time.time()

# # print(end - start)

#alex = Organism(1, 2, "test", 498, 402)
#alex.initialize_genome(50, 200)
#alex.save_polygons("test")

# alex.genome_to_array()
# alex.save_img()
# alex.save_img_vectorized()


# # alex.genome_to_array()
# # alex.calculate_fitness_mse(goal)
# # alex.save_img()
# # print(alex.fitness)

# # alex.genome_to_array_cairo()

# # alex.calculate_fitness_mse(goal)
# # alex.save_img()
# # print(alex.fitness)
