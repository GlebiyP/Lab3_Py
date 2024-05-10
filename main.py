import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
from scipy.spatial import ConvexHull

# Початкові координати сторони прямокутника
initial_rectangle_side = np.array([[0, 0], [0, 1]])

# Функція збереження напрямку сторони прямокутника
def save_rectangle_side_direction(points):
    direction = points[1] - points[0]
    return direction / np.linalg.norm(direction)

# Збереження напрямку сторони прямокутника
rectangle_side_direction = save_rectangle_side_direction(initial_rectangle_side)

def translate_points(points, translation):
    translated_points = points - translation
    return translated_points

def rotate_points(points, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points

def compress_to_square(points):
    max_dim = np.max(points, axis=0)
    min_dim = np.min(points, axis=0)
    center = (max_dim + min_dim) / 2
    max_side = np.max(max_dim - min_dim)
    compressed_points = (points - center) * (1 / max_side)
    return compressed_points

def find_furthest_points(points):
    max_dist = 0
    furthest_points = None, None
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist
                furthest_points = points[i], points[j]
    
    return furthest_points

def find_points_furthest_from_line(line, points):
    max_dist_left = 0
    max_dist_right = 0
    furthest_point_left = None
    furthest_point_right = None
    
    for point in points:
        dist = np.abs(np.cross(line[1]-line[0], point-line[0])) / np.linalg.norm(line[1]-line[0])
        
        # Перевіряємо, з якої сторони від прямої знаходиться точка
        side = np.sign(np.cross(line[1]-line[0], point-line[0]))
        
        if dist > max_dist_left and side > 0:
            max_dist_left = dist
            furthest_point_left = point
            
        if dist > max_dist_right and side < 0:
            max_dist_right = dist
            furthest_point_right = point
    
    return furthest_point_left, furthest_point_right

def draw_rectangle(points, line, parallel_lines, perpendicular_lines):
    plt.scatter(points[:,0], points[:,1], color='blue')
    plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='red')
    
    for parallel_line in parallel_lines:
        for i in range(len(parallel_line)-1):
            plt.plot([parallel_line[i][0], parallel_line[i+1][0]], [parallel_line[i][1], parallel_line[i+1][1]], color='green')
    
    for perpendicular_line in perpendicular_lines:
        for i in range(len(perpendicular_line)-1):
            plt.plot([perpendicular_line[i][0], perpendicular_line[i+1][0]], [perpendicular_line[i][1], perpendicular_line[i+1][1]], color='green')

    # Побудова опуклої оболонки
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Petunin Rectangle')
    plt.grid(True)
    plt.show()

def find_parallel_lines(line, points):
    parallel_lines = []

    for point in points:
        displacement = point - line[0]
        new_point = line[1] + displacement
        parallel_lines.append([point, new_point])

    return parallel_lines

def find_perpendicular_lines(line, furthest_points):
    perpendicular_lines = []

    # Знаходимо напрямок прямої
    direction = line[1] - line[0]
    slope = direction[1] / direction[0]
    
    # Знаходимо точки, через які треба провести перпендикуляри
    point1, point2 = furthest_points
    
    # Знаходимо крайні точки перпендикулярних прямих
    perpendicular_line1_point1 = point1
    perpendicular_line1_point2 = point1 + np.array([-direction[1], direction[0]])
    perpendicular_line2_point1 = point2
    perpendicular_line2_point2 = point2 + np.array([-direction[1], direction[0]])
    
    perpendicular_lines.append([perpendicular_line1_point1, perpendicular_line1_point2])
    perpendicular_lines.append([perpendicular_line2_point1, perpendicular_line2_point2])
    
    return perpendicular_lines

def find_square_center(points):
    # Знаходимо максимальне та мінімальне значення по кожній з осей
    max_dim = np.max(points, axis=0)
    min_dim = np.min(points, axis=0)
    
    # Знаходимо середнє значення між максимальним і мінімальним значеннями кожної осі
    center = (max_dim + min_dim) / 2
    
    return center

def distances_to_center(points, center):
    # Знаходимо відстані від кожної точки до центру квадрата
    distances = np.linalg.norm(points - center, axis=1)
    
    return distances

def inside_square(points):
    # Визначаємо, які точки знаходяться всередині квадрата
    center = find_square_center(points)
    max_dim = np.max(points, axis=0)
    min_dim = np.min(points, axis=0)
    inside_mask = (points[:, 0] >= min_dim[0]) & (points[:, 0] <= max_dim[0]) & \
                  (points[:, 1] >= min_dim[1]) & (points[:, 1] <= max_dim[1])
    return inside_mask

def plot_concentric_ellipses(center, ellipses):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # Побудова концентричних еліпсів
    for ellipse in ellipses:
        ax.add_patch(ellipse)

    # Встановлення відповідного масштабу вісей
    ax.autoscale()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Concentric Ellipses')
    plt.grid(True)
    plt.show()

def stretch_to_ellipses(distances, ratio):
    # Розтягуємо кожне коло у концентричний еліпс
    stretched_distances = distances * ratio
    
    return stretched_distances

def main():
    # Задаємо кількість точок
    num_points = 100
    
    # Генеруємо випадкові точки
    points = np.random.rand(num_points, 2) * 10  # Генеруємо в діапазоні [0, 10)

    # Знаходимо точки, що знаходяться всередині квадрата
    inside_points = points[inside_square(points)]
    
    # Знаходимо дві найвіддаленіші точки
    furthest_points = find_furthest_points(points)
    
    # Знаходимо пряму між найвіддаленішими точками
    line = furthest_points
    
    # Знаходимо по одній найвіддаленішій точці від прямої з кожної сторони
    left_point, right_point = find_points_furthest_from_line(line, points)
    
    # Знаходимо паралельні прямі, що проходять через ці точки
    parallel_lines = find_parallel_lines(line, [left_point, right_point])
    
    # Знаходимо перпендикулярні прямі, що проходять через найвіддаленіші точки
    perpendicular_lines = find_perpendicular_lines(line, furthest_points)

    # Знаходимо центр квадрата
    center_square = find_square_center(inside_points)
    
    # Збереження координат центра прямокутника
    initial_rectangle_center = center_square
    
    # Переносимо всі точки і прямокутник
    translation = left_point
    points = translate_points(points, translation)
    line = translate_points(line, translation)
    parallel_lines = [translate_points(parallel_line, translation) for parallel_line in parallel_lines]
    perpendicular_lines = [translate_points(perpendicular_line, translation) for perpendicular_line in perpendicular_lines]

    # Знаходимо кут нахилу прямокутника
    angle = np.arctan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
    
    # Повертаємо всі точки і прямокутник на протилежний кут
    points = rotate_points(points, -angle)
    line = rotate_points(line, -angle)
    parallel_lines = [rotate_points(parallel_line, -angle) for parallel_line in parallel_lines]
    perpendicular_lines = [rotate_points(perpendicular_line, -angle) for perpendicular_line in perpendicular_lines]

    # Знаходимо відношення розтягнення (ratio)
    max_dim = np.max(points, axis=0)
    min_dim = np.min(points, axis=0)
    ratio = max(max_dim - min_dim) / min(max_dim - min_dim)
    
    # Стискаємо прямокутник в квадрат
    points = compress_to_square(points)
    
    # Знаходимо центр квадрата
    center = find_square_center(inside_points)
    
    # Знаходимо відстані від центру квадрата до точок, що знаходяться всередині нього
    distances = distances_to_center(inside_points, center)

    # Розтягуємо кола в концентричні еліпси
    stretched_distances = stretch_to_ellipses(distances, ratio)

    ellipses = [Ellipse(center, distance * ratio, distance, color='red', fill=False) for distance in stretched_distances]
    
    # Повернення головної вісі еліпсів
    for ellipse in ellipses:
        ellipse.angle = np.degrees(-angle)

    # Перенесення еліпсів так, щоб їхній центр збігався із центром прямокутника Петуніна
    for ellipse in ellipses:
        ellipse_center = np.array([ellipse.center[0] + initial_rectangle_center[0], ellipse.center[1] + initial_rectangle_center[1]])
        ellipse.center = ellipse_center

    plot_concentric_ellipses(center, ellipses)

if __name__ == "__main__":
        main()
