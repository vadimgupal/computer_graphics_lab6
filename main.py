import numpy as np
from math import cos, sin, radians
import tkinter as tk
from tkinter import ttk

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
    """Поворот 3x3 вокруг единичной оси на угол в градусах."""
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
    """Икосаэдр, построенный с цилиндра.
    Полюса: (0,0,±sqrt(5)/2); кольца радиуса 1 на z=±1/2, нижнее смещено на 36°.
    Возвращает Polyhedron с явными 20 треугольными гранями.
    """
    def deg(a): return np.deg2rad(a)

    z_top, z_bot = +0.5, -0.5
    r = 1.0
    z_pole = np.sqrt(5.0) / 2.0

    V = []
    # верхняя вершина
    V.append((0.0, 0.0, +z_pole))

    # 1..5 — верхнее кольцо (углы 0,72,144,216,288)
    for k in range(5):
        ang = deg(72*k)
        V.append((r*np.cos(ang), r*np.sin(ang), z_top))

    # 6..10 — нижнее кольцо (углы 36,108,180,252,324)
    for k in range(5):
        ang = deg(36 + 72*k)
        V.append((r*np.cos(ang), r*np.sin(ang), z_bot))

    # нижняя вершина
    V.append((0.0, 0.0, -z_pole))

    F = []

    # Верхняя «шапка»: 5 треугольников (0, Ti, Ti+1)
    for i in range(5):
        F.append([0, 1+i, 1+((i+1) % 5)])

    # Средняя зона: 10 треугольников (по 2 на «сектор»).
    # Важный момент: у вершины верхнего кольца Ti ближайшие нижние — Bi и B(i-1).
    for i in range(5):
        Ti   = 1 + i
        Tip1 = 1 + ((i+1) % 5)
        Bi   = 6 + i
        Bim1 = 6 + ((i-1) % 5)

        # «верхний» из пары (Ti, Bi, B(i-1))
        F.append([Ti, Bi, Bim1])
        # «нижний» из пары (Bi, Tip1, Ti)
        F.append([Bi, Tip1, Ti])

    # Нижняя «шапка»: 5 треугольников (11, Bj+1, Bj)
    for j in range(5):
        Bj   = 6 + j
        Bjp1 = 6 + ((j+1) % 5)
        F.append([11, Bjp1, Bj])

    return Polyhedron(V, F)


def dodecahedron():
    """Додекаэдр как дуал к идущему выше 'цилиндрическому' икосаэдру:
    вершины = центры тяжести треугольных граней икосаэдра,
    грани = пятиугольники, по одному на каждую вершину икосаэдра.
    """
    I = icosahedron()
    # координаты вершин икосаэдра (N x 3)
    V = (I.V[:3, :] / I.V[3, :]).T #normalization
    faces_I = [f.indices for f in I.faces]  # список где каждый элемент это список индексов точек грани

    # 20 вершин додекаэдра: центроиды треугольников
    D_vertices = [tuple(np.mean(V[idxs], axis=0)) for idxs in faces_I]

    # каждой вершине икосаэдра ставим в соответствие номера граней в которых она используется
    incident = [[] for _ in range(len(V))]
    for fi, tri in enumerate(faces_I):
        for vid in tri:
            incident[vid].append(fi)

    # Построим 12 пятиугольных граней додекаэдра.
    D_faces = []
    for vid, fids in enumerate(incident):
        if len(fids) != 5:
            # на всякий случай пропустим аномалии (их быть не должно)
            continue
        p = V[vid]              # точка-центр «звезды» (вершины икосаэдра)
        n = normalize(p)        # используем направление p как «нормаль» локальной плоскости
        # ортонормированный базис {e1,e2} в плоскости, перпендикулярной n
        tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = normalize(np.cross(n, tmp))
        e2 = np.cross(n, e1)

        # отсортируем прилегающие центроиды по углу в этой плоскости
        ang_with_id = []
        for fid in fids:
            c = np.mean(V[faces_I[fid]], axis=0)
            d = c - p
            ang = np.arctan2(np.dot(d, e2), np.dot(d, e1))
            ang_with_id.append((ang, fid))
        ang_with_id.sort()#массив отсортированных по полярному углу точек, чтобы точки брались по кругу

        D_faces.append([fid for ang, fid in ang_with_id]) #добавляем грань

    return Polyhedron(D_vertices, D_faces)


# --------------------
# Вспомогательные функции для визуализации (2D каркас после проекции)
# --------------------

# --------------------
# Tkinter-приложение для отображения каркаса
# --------------------

POLY_BUILDERS = {
    'Тетраэдр': tetrahedron,
    'Гексаэдр (куб)': hexahedron,
    'Октаэдр': octahedron,
    'Икосаэдр': icosahedron,
    'Додекаэдр': dodecahedron,
}

def make_poly(name: str) -> Polyhedron:
    """Создаёт выбранный многогранник без дополнительных поворотов/масштабов."""
    builder = POLY_BUILDERS.get(name)
    if builder is None:
        builder = hexahedron
    return builder()

def project_points(P: Polyhedron, proj_mode: str, f: float = 1.8):
    """Возвращает 2D проекцию вершин и список рёбер.

    proj_mode: 'perspective' или 'axonometric'/'isometric'
    """
    Q = P.copy()
    if proj_mode == 'perspective':
        # Стандартная перспектива: камера в начале координат, смотрит вдоль +Z; сместим модель на z=5
        Q = Q.translate(0, 0, 5.0)
        M = perspective(f)
    else:
        # Стандартная изометрическая (аксонометрическая) матрица
        M = isometric_projection_matrix()
    x, y = Q.projected(M)
    edges = Q.edges()
    return (x, y, edges)

def to_pixels(x, y, width, height, scale=120.0):
    """Перевод модельных координат в пиксели с фиксированным масштабом и центрированием."""
    x = np.asarray(x)
    y = np.asarray(y)
    cx = width * 0.5
    cy = height * 0.5
    Xs = cx + scale * x
    Ys = cy - scale * y  # переворот оси Y
    return Xs, Ys

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Правильные многогранники — Tkinter')

        self.poly_var = tk.StringVar(value='Гексаэдр (куб)')
        self.proj_var = tk.StringVar(value='perspective')  # 'perspective' | 'isometric'
        # Текущая модель многогранника
        self.model: Polyhedron = make_poly(self.poly_var.get())

        top = ttk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text='Многогранник:').pack(side=tk.LEFT)
        self.poly_box = ttk.Combobox(
            top,
            textvariable=self.poly_var,
            values=list(POLY_BUILDERS.keys()),
            state='readonly',
            width=18,
        )
        self.poly_box.pack(side=tk.LEFT, padx=(6, 12))
        self.poly_box.bind('<<ComboboxSelected>>', lambda e: self.rebuild_model())

        ttk.Label(top, text='Проекция:').pack(side=tk.LEFT)
        self.rb_persp = ttk.Radiobutton(
            top, text='Перспективная', value='perspective', variable=self.proj_var,
            command=self.redraw
        )
        self.rb_iso = ttk.Radiobutton(
            top, text='Аксонометрическая', value='isometric', variable=self.proj_var,
            command=self.redraw
        )
        self.rb_persp.pack(side=tk.LEFT, padx=(6, 6))
        self.rb_iso.pack(side=tk.LEFT)

        ttk.Button(top, text='Сброс', command=self.rebuild_model).pack(side=tk.RIGHT)

        # Панель преобразований
        controls = ttk.LabelFrame(root, text='Преобразования')
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,8))

        # Смещение
        trf1 = ttk.Frame(controls)
        trf1.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf1, text='Смещение: dx').pack(side=tk.LEFT)
        self.dx_entry = ttk.Entry(trf1, width=6)
        self.dx_entry.insert(0, '0')
        self.dx_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf1, text='dy').pack(side=tk.LEFT)
        self.dy_entry = ttk.Entry(trf1, width=6)
        self.dy_entry.insert(0, '0')
        self.dy_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf1, text='dz').pack(side=tk.LEFT)
        self.dz_entry = ttk.Entry(trf1, width=6)
        self.dz_entry.insert(0, '0')
        self.dz_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Button(trf1, text='Применить', command=self.apply_translate).pack(side=tk.LEFT, padx=6)

        # Поворот
        trf2 = ttk.Frame(controls)
        trf2.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf2, text='Поворот: ось').pack(side=tk.LEFT)
        self.rot_axis_var = tk.StringVar(value='x')
        self.rot_axis = ttk.Combobox(trf2, textvariable=self.rot_axis_var, values=['x','y','z'], state='readonly', width=4)
        self.rot_axis.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf2, text='угол (°)').pack(side=tk.LEFT)
        self.angle_entry = ttk.Entry(trf2, width=8)
        self.angle_entry.insert(0, '30')
        self.angle_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Button(trf2, text='Повернуть', command=self.apply_rotate).pack(side=tk.LEFT, padx=6)
        ttk.Button(trf2, text='Повернуть (через центр)', command=self.apply_rotate_center).pack(side=tk.LEFT, padx=6)

        # Масштаб
        trf3 = ttk.Frame(controls)
        trf3.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf3, text='Масштаб: sx').pack(side=tk.LEFT)
        self.sx_entry = ttk.Entry(trf3, width=6)
        self.sx_entry.insert(0, '1')
        self.sx_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf3, text='sy').pack(side=tk.LEFT)
        self.sy_entry = ttk.Entry(trf3, width=6)
        self.sy_entry.insert(0, '1')
        self.sy_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf3, text='sz').pack(side=tk.LEFT)
        self.sz_entry = ttk.Entry(trf3, width=6)
        self.sz_entry.insert(0, '1')
        self.sz_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Button(trf3, text='Масштаб', command=self.apply_scale).pack(side=tk.LEFT, padx=6)

        # Равномерный масштаб вокруг центра (одно число)
        ttk.Label(trf3, text=' s').pack(side=tk.LEFT, padx=(12,2))
        self.s_uniform_entry = ttk.Entry(trf3, width=6)
        self.s_uniform_entry.insert(0, '1')
        self.s_uniform_entry.pack(side=tk.LEFT, padx=(2,6))
        ttk.Button(trf3, text='Масштаб (через центр)', command=self.apply_scale_center).pack(side=tk.LEFT, padx=6)

        # Вращение вокруг произвольной прямой (p1 -> p2)
        trf5 = ttk.Frame(controls)
        trf5.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf5, text='Поворот вокруг прямой:').pack(side=tk.LEFT)
        ttk.Label(trf5, text='p1(x,y,z)').pack(side=tk.LEFT, padx=(8,2))
        self.p1x_entry = ttk.Entry(trf5, width=5); self.p1x_entry.insert(0,'0'); self.p1x_entry.pack(side=tk.LEFT)
        self.p1y_entry = ttk.Entry(trf5, width=5); self.p1y_entry.insert(0,'0'); self.p1y_entry.pack(side=tk.LEFT)
        self.p1z_entry = ttk.Entry(trf5, width=5); self.p1z_entry.insert(0,'0'); self.p1z_entry.pack(side=tk.LEFT)
        ttk.Label(trf5, text='p2(x,y,z)').pack(side=tk.LEFT, padx=(8,2))
        self.p2x_entry = ttk.Entry(trf5, width=5); self.p2x_entry.insert(0,'0'); self.p2x_entry.pack(side=tk.LEFT)
        self.p2y_entry = ttk.Entry(trf5, width=5); self.p2y_entry.insert(0,'1'); self.p2y_entry.pack(side=tk.LEFT)
        self.p2z_entry = ttk.Entry(trf5, width=5); self.p2z_entry.insert(0,'0'); self.p2z_entry.pack(side=tk.LEFT)
        ttk.Label(trf5, text='угол (°)').pack(side=tk.LEFT, padx=(8,2))
        self.angle_line_entry = ttk.Entry(trf5, width=7); self.angle_line_entry.insert(0,'30'); self.angle_line_entry.pack(side=tk.LEFT)
        ttk.Button(trf5, text='Повернуть (линия)', command=self.apply_rotate_line).pack(side=tk.LEFT, padx=6)

        # Отражение
        trf4 = ttk.Frame(controls)
        trf4.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf4, text='Отражение: плоскость').pack(side=tk.LEFT)
        self.refl_plane_var = tk.StringVar(value='xy')
        self.refl_plane = ttk.Combobox(trf4, textvariable=self.refl_plane_var, values=['xy','yz','xz'], state='readonly', width=6)
        self.refl_plane.pack(side=tk.LEFT, padx=(6,10))
        ttk.Button(trf4, text='Отразить', command=self.apply_reflect).pack(side=tk.LEFT, padx=6)

        self.canvas = tk.Canvas(root, bg='white', width=800, height=600)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda e: self.redraw())

        self.redraw()

    def get_poly(self) -> Polyhedron:
        return self.model

    def rebuild_model(self):
        self.model = make_poly(self.poly_var.get())
        self.redraw()

    def _parse_float(self, widget, default=0.0):
        try:
            return float(widget.get())
        except Exception:
            return default

    def apply_translate(self):
        dx = self._parse_float(self.dx_entry, 0.0)
        dy = self._parse_float(self.dy_entry, 0.0)
        dz = self._parse_float(self.dz_entry, 0.0)
        self.model.translate(dx, dy, dz)
        self.redraw()

    def apply_rotate(self):
        axis = (self.rot_axis_var.get() or 'x').lower()
        angle = self._parse_float(self.angle_entry, 0.0)
        # Поворот вокруг оси, проходящей через начало координат
        if axis == 'x':
            self.model.rotate_x(angle)
        elif axis == 'y':
            self.model.rotate_y(angle)
        else:
            self.model.rotate_z(angle)
        self.redraw()

    def apply_rotate_center(self):
        """Явное вращение вокруг прямой через центр модели, параллельной выбранной оси."""
        axis = (self.rot_axis_var.get() or 'x').lower()
        angle = self._parse_float(self.angle_entry, 0.0)
        self.model.rotate_around_axis_through_center(axis, angle)
        self.redraw()

    def apply_scale(self):
        sx = self._parse_float(self.sx_entry, 1.0)
        sy = self._parse_float(self.sy_entry, 1.0)
        sz = self._parse_float(self.sz_entry, 1.0)
        # Анизотропный масштаб вокруг начала координат
        self.model.scale(sx, sy, sz)
        self.redraw()

    def apply_scale_center(self):
        # Равномерный масштаб вокруг центра модели (одно число)
        s = self._parse_float(self.s_uniform_entry, 1.0)
        self.model.scale_about_center(s)
        self.redraw()

    def apply_reflect(self):
        plane = (self.refl_plane_var.get() or 'xy').lower()
        if plane not in ('xy','yz','xz'):
            plane = 'xy'
        self.model.reflect(plane)
        self.redraw()

    def apply_rotate_line(self):
        # Чтение точек p1, p2 и угла
        x1 = self._parse_float(self.p1x_entry, 0.0)
        y1 = self._parse_float(self.p1y_entry, 0.0)
        z1 = self._parse_float(self.p1z_entry, 0.0)
        x2 = self._parse_float(self.p2x_entry, 0.0)
        y2 = self._parse_float(self.p2y_entry, 1.0)
        z2 = self._parse_float(self.p2z_entry, 0.0)
        angle = self._parse_float(self.angle_line_entry, 0.0)
        p1 = (x1, y1, z1)
        p2 = (x2, y2, z2)
        # Проверка на нулевую ось
        if np.linalg.norm(np.asarray(p2, float) - np.asarray(p1, float)) < 1e-12:
            # Ничего не делаем, если ось нулевая
            return
        self.model.rotate_around_line(p1, p2, angle)
        self.redraw()

    def redraw(self):
        self.canvas.delete('all')
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 10 or H < 10:
            return

        P = self.get_poly()
        mode = self.proj_var.get()
        x, y, edges = project_points(P, mode)
        # Фиксированный небольшой размер фигур
        Xs, Ys = to_pixels(x, y, W, H, scale=120.0)

        # Рисуем рёбра
        for a, b in edges:
            self.canvas.create_line(float(Xs[a]), float(Ys[a]), float(Xs[b]), float(Ys[b]), fill='#1f77b4')


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
