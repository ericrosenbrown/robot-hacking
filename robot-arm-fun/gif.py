import math
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

def forward_kinematics(l1=1,l2=1,theta=(0,0)):
    x1 = math.cos(theta[0])*l1
    y1= math.sin(theta[0])*l1
    
    x2 = x1 + math.cos(theta[1]+theta[0])*l2 
    y2 = y1 + math.sin(theta[1]+theta[0])*l2 
    
    return((x1,y1),(x2,y2))

def visualize_robot(l1=1,l2=1,theta=(0,0),show=True,print_link=False,colors=("red","blue")):
    link1, link2 = forward_kinematics(l1,l2,theta)
    if print_link:
        print(f"link1: {link1}, link2:{link2}")
    plt.plot(*zip((0,0),link1), marker='o',color=colors[0])
    plt.plot(*zip(link1,link2), marker='o',color=colors[1])
    
    plt.xlim([-(l1+l2+1), l1+l2+1])
    plt.ylim([-(l1+l2+1), l1+l2+1])
    if show:
        plt.show()

def _J(theta=[0,0],l1=1,l2=1):
    dx_t0 = -math.sin(theta[0])*l1 - math.sin(theta[0]+theta[1])*l2
    dx_t1 = - math.sin(theta[0]+theta[1])*l2
    dy_t0 = math.cos(theta[0])*l1 + math.cos(theta[0] + theta[1])*l2
    dy_t1 = math.cos(theta[0] + theta[1])*l2
    return(np.array([[dx_t0, dx_t1],[dy_t0, dy_t1]]))

theta = [math.pi/4, math.pi/2]
delta = 0.05
radius = 1.25 #The radius of the circle we intended to draw
#Question: In order to make drawn circle be this radius, should I somehow relate delta to radius?
num_steps = 50

for _ in range(1):
    past_points = []
    cur_degree = 0 #this is the current degrees in the circle we are drawing, start at 0
    counter = 0
    for cur_degree in np.arange(0,2*math.pi,2*math.pi/num_steps):

        _, link2 = forward_kinematics(theta=theta)
        past_points.append(link2)

        visualize_robot(theta=theta,show=False)

        for (x,y) in past_points:
            plt.scatter(x,y,color="black")

        #claculate the tangent to the circle at this degree. The tangent defines how we want the end effector tip to move
        x_dot = -math.sin(cur_degree)*radius
        y_dot = math.cos(cur_degree)*radius
        
        tau_dot = [x_dot,y_dot]
        
        J = _J(theta=(theta))

        J_inv = np.linalg.inv(J)

        theta_dot = np.matmul(J_inv, tau_dot)
        theta = theta + theta_dot * delta

        plt.draw()
        plt.savefig(f'gif-images/{counter}.png')
        plt.pause(0.0001)
        plt.clf()

        counter+=1
    
# Build GIF
dir_path = "./gif-images/"
with imageio.get_writer('robot.gif', mode='I',fps=20) as writer:
    for filename in range(num_steps):
        filename = str(filename)
        image = imageio.imread(dir_path+filename+".png")
        writer.append_data(image)
        