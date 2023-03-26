from PIL import Image
import numpy as np
import turtle

# cnames = {
# 'dodgerblue':           '#1E90FF',
# 'orange':               '#FFA500',
# 'green':                '#008000',
# 'red':                  '#FF0000',
# 'mediumslateblue':      '#7B68EE',
# 'saddlebrown':          '#8B4513',
# 'violet':               '#EE82EE',
# 'dimgrey':              '#696969',
# 'darkkhaki':            '#BDB76B',
# 'turquoise':            '#40E0D0',}

# This does not violate our random selection of colors, 
# We set this up just to make sure that the color of the
# other shapes (i.e., square and prismatic) matches the random color of the circle
color_list = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'mediumpurple', 'saddlebrown',
              'violet', 'gray', 'y', 'darkturquoise']


import matplotlib.pyplot as plt
def circle():
    a, b = (0., 0.)
    theta = np.arange(0, 2 * np.pi, 0.01)
    # theta[-1] = 2 * np.pi
    plt.figure(figsize=(6,6))
    # 6x6-->7  0 250 20
    # 3x3-->3.5
    for r in range(10, 250, 20):
        x = a + r * np.cos(theta)
        y = b + r * np.sin(theta)
        plt.plot(x, y, linewidth = 6.75)
    plt.axis('off')
    plt.savefig('circle.png')
    plt.show()

def circle_density(d):
    a, b = (0., 0.)
    theta = np.arange(0, 2 * np.pi, 0.01)
    # theta[-1] = 2 * np.pi
    plt.figure(figsize=(6,6))
    # 6x6-->7  0 250 20
    # 3x3-->3.5
    for r in range(10, d, 20):
        x = a + r * np.cos(theta)
        y = b + r * np.sin(theta)
        plt.plot(x, y, linewidth = 6.75)
    plt.axis('off')
    plt.savefig('density/circle{}.png'.format(str(d)))
    # plt.show()


def square(color_list, dense):
    i = 0
    plt.figure(figsize=(6,6))
    for b in range(10, dense, 20):
        x = np.arange(-b, b, 0.01)
        y1 = [b] * len(x)
        y2 = [-b] * len(x)

        y = np.arange(-b, b, 0.01)
        x1 = [-b] * len(y)
        x2 = [b] * len(y)

        plt.plot(x, y1, linewidth = 6.75, c = color_list[i])
        plt.plot(x, y2, linewidth = 6.75, c = color_list[i])
        plt.plot(x1, y, linewidth = 6.75, c = color_list[i])
        plt.plot(x2, y, linewidth = 6.75, c = color_list[i])

        i += 1
        i = i % 10
    plt.axis('off')
    plt.savefig('density/squre{}.png'.format(dense))
    # plt.show()


def prismatic(color_list, dense):
    i = 0
    plt.figure(figsize=(6,6))
    for b in range(10, dense, 20):
        b = b * 1.414
        x = np.arange(-b, b, 0.005)
        # print(x.min())
        # print(x)
        y1 = b - abs(x)
        y2 = -b + abs(x)
        plt.plot(x, y1, linewidth = 6.75, c = color_list[i])
        plt.plot(x, y2, linewidth = 6.75, c = color_list[i])

        i += 1
        i = i % 10

    plt.axis('off')
    plt.savefig('density/rhom{}.png'.format(dense))
    # plt.show()


def square2rhom(img_pth, d):
    img = Image.open(img_pth).convert('RGB')
    img = img.rotate(angle=45, resample=Image.BICUBIC, fillcolor = (255,255,255))
    # plt.show()
    # exit()
    img.save('rhom{}.png'.format(d), quality=95)


if __name__ == '__main__':
    circle()



