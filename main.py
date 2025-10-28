import numpy as np
from math import cos, sin, radians
import matplotlib.pyplot as plt

def to_h(point3):
    """Возвращает однородный 4x1 вектор из 3D точки (x, y, z)."""
    x, y, z = point3
    return np.array([x, y, z, 1.0], dtype=float)

def from_h(vec4):
    """Возвращает 3D точку из однородного вектора после перспективного деления."""
    w = vec4[3]
    if w == 0:
        raise ZeroDivisionError("Однородная координата w == 0 при дегомогенизации")
    return (vec4[:3] / w)

def normalize(v):
    """Нормализует вектор."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

# --------------------
# Матрицы преобразований (4x4, вектор-столбцы)
# --------------------

def T(dx, dy, dz):
    """Матрица переноса (смещения)."""
    M = np.eye(4)
    M[:3, 3] = [dx, dy, dz]
    return M

def S(sx, sy, sz):
    """Матрица масштабирования."""
    M = np.eye(4)
    M[0,0], M[1,1], M[2,2] = sx, sy, sz
    return M

def Rx(angle_deg):
    """Матрица поворота вокруг оси X."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[1,1], M[1,2] = ca, -sa
    M[2,1], M[2,2] = sa,  ca
    return M

def Ry(angle_deg):
    """Матрица поворота вокруг оси Y."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[0,0], M[0,2] =  ca, sa
    M[2,0], M[2,2] = -sa, ca
    return M

def Rz(angle_deg):
    """Матрица поворота вокруг оси Z."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[0,0], M[0,1] = ca, -sa
    M[1,0], M[1,1] = sa,  ca
    return M

def reflect(plane: str):
    """Отражение относительно координатной плоскости: 'xy', 'yz', или 'xz'."""
    plane = plane.lower()
    if plane == "xy":
        return S(1, 1, -1)
    if plane == "yz":
        return S(-1, 1, 1)
    if plane == "xz":
        return S(1, -1, 1)
    raise ValueError("Плоскость должна быть: 'xy', 'yz', 'xz'")

def rodrigues_axis_angle(axis, angle_deg):
    """Поворот 3x3 (формула Родрига) вокруг единичной оси на угол в градусах."""
    axis = normalize(np.asarray(axis, dtype=float))
    a = radians(angle_deg)
    c, s = cos(a), sin(a)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3)*c + (1-c)*np.outer(axis, axis) + s*K
    return R

def R_around_line(p1, p2, angle_deg):
    """Матрица поворота 4x4 вокруг произвольной 3D линии p1->p2 на угол."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    axis = p2 - p1
    R3 = rodrigues_axis_angle(axis, angle_deg)  # 3x3
    M = np.eye(4)
    M[:3,:3] = R3
    # Сэндвич-преобразование для поворота вокруг линии, проходящей через p1
    return T(*p1) @ M @ T(*(-p1))

# --------------------
# Матрицы проекций
# --------------------

def perspective(f=1.5):
    """
    Простая матрица перспективной проекции.
    Камера в начале координат смотрит вдоль +Z; точки сцены должны иметь z > 0.
    """
    M = np.eye(4)
    M[3,2] = 1.0 / f  # w' = z/f  -> x' = x / (z/f) = f*x/z
    return M

def ortho_xy():
    """Ортографическая проекция на плоскость XY (отбрасывание Z)."""
    M = np.eye(4)
    M[2,2] = 0.0
    return M

def isometric_projection_matrix():
    """Аксонометрическая (изометрическая) проекция = поворот + ортографическая проекция."""
    # Классическая изометрия: поворот вокруг Z на 45°, затем вокруг X на ~35.264°
    alpha = 35.264389682754654
    beta = 45.0
    R = Rx(alpha) @ Rz(beta)
    return ortho_xy() @ R

# --------------------
# Геометрические классы
# --------------------

class Point:
    """Класс для представления точки в 3D пространстве."""
    def __init__(self, x, y, z):
        self.v = to_h((x, y, z))

    @property
    def xyz(self):
        return from_h(self.v)

    def as_array(self):
        return self.v.copy()

class PolygonFace:
    """Класс для представления грани многогранника."""
    def __init__(self, vertex_indices):
        self.indices = list(vertex_indices)

class Polyhedron:
    """Класс для представления многогранника."""
    def __init__(self, vertices, faces):
        """
        vertices: список вершин (кортежи (x,y,z))
        faces: список граней (списки индексов вершин)
        """
        self.V = np.array([to_h(p) for p in vertices], dtype=float).T  # 4xN (столбец = вершина)
        self.faces = [PolygonFace(f) for f in faces]

    # --- основные методы ---
    def copy(self):
        """Создает копию многогранника."""
        P = Polyhedron([(0,0,0)], [[]])
        P.V = self.V.copy()
        P.faces = [PolygonFace(f.indices.copy()) for f in self.faces]
        return P

    def center(self):
        """Вычисляет центр многогранника."""
        pts = self.V[:3, :] / self.V[3, :]
        return np.mean(pts, axis=1)

    def apply(self, M):
        """Применяет матричное преобразование 4x4."""
        self.V = M @ self.V
        return self

    # --- удобные методы преобразований (все через матрицы) ---
    def translate(self, dx, dy, dz):
        """Перенос (смещение)."""
        return self.apply(T(dx, dy, dz))

    def scale(self, sx, sy, sz):
        """Масштабирование."""
        return self.apply(S(sx, sy, sz))

    def scale_about_center(self, s):
        """Масштабирование относительно центра."""
        c = self.center()
        return self.apply(T(*(-c)) @ S(s, s, s) @ T(*c))

    def rotate_x(self, angle_deg):
        """Поворот вокруг оси X."""
        return self.apply(Rx(angle_deg))

    def rotate_y(self, angle_deg):
        """Поворот вокруг оси Y."""
        return self.apply(Ry(angle_deg))

    def rotate_z(self, angle_deg):
        """Поворот вокруг оси Z."""
        return self.apply(Rz(angle_deg))

    def reflect(self, plane: str):
        """Отражение относительно координатной плоскости."""
        return self.apply(reflect(plane))

    def rotate_around_axis_through_center(self, axis: str, angle_deg):
        """Поворот вокруг оси, проходящей через центр."""
        axis = axis.lower()
        c = self.center()
        R = {'x': Rx, 'y': Ry, 'z': Rz}[axis](angle_deg)
        return self.apply(T(*(-c)) @ R @ T(*c))

    def rotate_around_line(self, p1, p2, angle_deg):
        """Поворот вокруг произвольной линии."""
        return self.apply(R_around_line(p1, p2, angle_deg))

    # --- вспомогательные методы для ребер ---
    def edges(self):
        """Вычисляет список ребер многогранника."""
        es = set()
        if self.faces and len(self.faces[0].indices)>0:
            # Строим ребра из граней
            for f in self.faces:
                idx = f.indices
                for i in range(len(idx)):
                    a = idx[i]
                    b = idx[(i+1) % len(idx)]
                    es.add(tuple(sorted((a,b))))
        else:
            # Резервный метод: соединение ближайших соседей
            pts = (self.V[:3,:] / self.V[3,:]).T
            n = len(pts)
            D = np.linalg.norm(pts[None,:,:]-pts[:,None,:], axis=-1)
            for i in range(n):
                neigh = list(np.argsort(D[i])[1:4])  # 3 ближайших соседа
                for j in neigh:
                    es.add(tuple(sorted((i,j))))
        return sorted(list(es))

    # --- проекция ---
    def projected(self, matrix4x4):
        """Возвращает 2D точки (x,y) после применения матрицы проекции."""
        Pv = matrix4x4 @ self.V
        # Перспективное деление
        Pv = Pv / Pv[3, :]
        # возвращаем только (x,y)
        return Pv[0, :], Pv[1, :]

# --------------------
# Правильные многогранники (Платоновы тела)
# --------------------

def tetrahedron():
    """Создает правильный тетраэдр с центром в начале координат."""
    V = [(1, 1, 1),
         (1,-1,-1),
         (-1, 1,-1),
         (-1,-1, 1)]
    F = [
        [0,1,2],
        [0,3,1],
        [0,2,3],
        [1,3,2]
    ]
    return Polyhedron(V, F)

def hexahedron():
    """Правильный гексаэдр (куб) с центром в начале координат и ребром 2."""
    V = [
        (-1, -1, -1),  # 0
        ( 1, -1, -1),  # 1
        ( 1,  1, -1),  # 2
        (-1,  1, -1),  # 3
        (-1, -1,  1),  # 4
        ( 1, -1,  1),  # 5
        ( 1,  1,  1),  # 6
        (-1,  1,  1),  # 7
    ]
    # 6 квадратных граней (порядок вершин по контуру)
    F = [
        [0, 1, 2, 3],  # z = -1 (низ)
        [4, 5, 6, 7],  # z = +1 (верх)
        [0, 1, 5, 4],  # y = -1
        [1, 2, 6, 5],  # x = +1
        [2, 3, 7, 6],  # y = +1
        [3, 0, 4, 7],  # x = -1
    ]
    return Polyhedron(V, F)

def octahedron():
    """Правильный октаэдр с центром в начале координат и ребром √2."""
    V = [
        ( 1,  0,  0),  # 0
        (-1,  0,  0),  # 1
        ( 0,  1,  0),  # 2
        ( 0, -1,  0),  # 3
        ( 0,  0,  1),  # 4 (верх)
        ( 0,  0, -1),  # 5 (низ)
    ]
    # 8 треугольных граней
    F = [
        [4, 0, 2],
        [4, 2, 1],
        [4, 1, 3],
        [4, 3, 0],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3],
    ]
    return Polyhedron(V, F)

def icosahedron():
    """Создает правильный икосаэдр."""
    phi = (1 + 5 ** 0.5) / 2  # золотое сечение
    a, b = 1.0, phi
    V = []
    for x in (-a, a):
        for y in (-b, b):
            V.append((x, y, 0))
            V.append((0, x, y))
            V.append((y, 0, x))
    # Удаляем дубликаты
    V_unique = []
    for p in V:
        if not any(np.allclose(p, q) for q in V_unique):
            V_unique.append(p)
    V = V_unique
    # Строим грани через поиск треугольников с равными ребрами
    pts = np.array(V)
    n = len(pts)
    dists = {}
    for i in range(n):
        for j in range(i+1, n):
            dists[(i,j)] = np.linalg.norm(pts[i]-pts[j])
    edges_sorted = sorted(dists.items(), key=lambda kv: kv[1])
    E = set(k for k,_ in edges_sorted[:30])
    F = set()
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) not in E: continue
            for k in range(j+1, n):
                if (i,k) in E and (j,k) in E:
                    F.add(tuple([i,j,k]))
    return Polyhedron([tuple(p) for p in pts], [list(f) for f in sorted(F)])

def dodecahedron():
    """Создает правильный додекаэдр."""
    phi = (1 + 5 ** 0.5) / 2  # золотое сечение
    b = 1.0/phi
    c = phi
    V = []
    # (±1, ±1, ±1)
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            for z in (-1.0, 1.0):
                V.append((x,y,z))
    # (0, ±1/φ, ±φ) и перестановки
    for sx in (1.0,-1.0):
        for sy in (1.0,-1.0):
            V.append((0.0, sx*b, sy*c))
            V.append((sx*b, sy*c, 0.0))
            V.append((sx*c, 0.0, sy*b))
    # Грани - 12 пятиугольников; для простоты используем ребра ближайших соседей
    return Polyhedron(V, [])

# --------------------
# Вспомогательные функции для визуализации (2D каркас после проекции)
# --------------------

def wireframe_2d(ax, P: Polyhedron, proj='perspective', f=1.5):
    """Отрисовывает многогранник в 2D после проекции."""
    if proj == 'perspective':
        M = perspective(f)
    elif proj == 'axonometric' or proj == 'isometric':
        M = isometric_projection_matrix()
    else:
        raise ValueError("proj должна быть 'perspective' или 'axonometric'")
    x, y = P.projected(M)
    # рисуем ребра
    for a,b in P.edges():
        ax.plot([x[a], x[b]], [y[a], y[b]])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)

def demo():
    """Демонстрационная функция — отображает 5 правильных многогранников."""
    TET = tetrahedron().scale_about_center(1.0).translate(-3.0,  1.2, 7)
    CUB = hexahedron().scale_about_center(0.9).rotate_around_axis_through_center('z', 30).translate(0.0,  1.2, 7)
    OCT = octahedron().scale_about_center(1.1).rotate_around_axis_through_center('x', 25).translate(3.0,  1.2, 7)

    ICO = icosahedron().scale_about_center(0.9).rotate_around_axis_through_center('x', 20).translate(-1.5, -1.2, 7)
    DOD = dodecahedron().scale_about_center(0.9).rotate_around_axis_through_center('z', 30).translate(1.5, -1.2, 7)

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2,3,1)
    wireframe_2d(ax1, TET, proj='perspective', f=2.0)
    ax1.set_title('Тетраэдр — перспективная')

    ax2 = fig.add_subplot(2,3,2)
    wireframe_2d(ax2, CUB, proj='axonometric')  # куб в изометрии красиво читается
    ax2.set_title('Гексаэдр (куб) — аксонометрическая')

    ax3 = fig.add_subplot(2,3,3)
    wireframe_2d(ax3, OCT, proj='perspective', f=2.0)
    ax3.set_title('Октаэдр — перспективная')

    ax4 = fig.add_subplot(2,3,4)
    wireframe_2d(ax4, ICO, proj='axonometric')
    ax4.set_title('Икосаэдр — аксонометрическая')

    ax5 = fig.add_subplot(2,3,5)
    wireframe_2d(ax5, DOD, proj='perspective', f=2.0)
    ax5.set_title('Додекаэдр — перспективная')

    # Пустую шестую ячейку можно скрыть
    ax6 = fig.add_subplot(2,3,6)
    ax6.axis('off')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    demo()
