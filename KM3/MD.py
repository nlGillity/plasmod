from typing import Optional
import numpy as np
from scipy.constants import electron_mass, e, epsilon_0, pi, c
import matplotlib.pyplot as plt
from matplotlib import animation
from progress.bar import IncrementalBar

"""
    Масштабные величины:
"""
print("Парамерты системы:")
# Начальная энергия частицы (Дж)
E0 = 5 * e  
print("E0 = {0:.4e} Дж".format(E0))
# Масса частицы (кг)
m = electron_mass
print("m = {0:.4e} кг".format(m))
# Масштаб по пространству (м)
l = 10**-10
print(f"l = {l:.2e} м")
# Модуль макс. скорости частицы
v_max = np.sqrt(2 * E0 / m)
print(f"v_max = {v_max:.2e} м/с")
# Масштаб по времени (с)
t = l / v_max
print(f"t = {t:.2e} с")

# Масштабный коэффициент
k = (4 * pi * epsilon_0)**-1  # (В * нм**2 / Кл**2)
W = k / m / l**3 * t**2 * e**2  
print(f"Масштабный коэффициент W = {W:.4e}")

"""
    Безразмерные величины:
"""
M  = 1  # масса частицы
N  = 3  # кол-во частиц
L  = 1  # размер куба
T_max = 10     # максимальное время расчета
H     = 0.01  # шаг по времени
N_t   = int(np.floor(T_max / H)) # количество узлов во временнной сетке
 
"""
    Начальные условия:
"""
# Разыгровка начальных координат частиц
R0 = np.random.uniform(-L / 2, L / 2, (N, 1, 3))
# Разыгровка начальных скоростей частиц 
V0 = np.random.uniform(-1, 1, (N, 1, 3)) 
# Коррекция начальных скоростей частиц         
P_sum = np.sum(m * V0, axis = 0)[0]
V0 -= P_sum / (N * m)

# Определение начальных энергий частиц (эВ)
print("\nЭнергии, разыгранные между частицами:")
average = 0
for i in range(N):
    v = np.linalg.norm(V0[i]) * v_max  # модуль размерной скорости
    energy = m * v**2 / 2  # размерная энергия (Дж)
    energy /= e  # энергия в эВ
    print(f"  {i + 1}-я частица: {energy:.2f} эВ / {M * np.linalg.norm(V0[i])**2 / 2:.2f} (v = {v:.2e} м/c / {v / v_max:.3f})")
    average += energy
average /= N
print(f"Средняя энергия частиц: {average:.2f} эВ\n")

"""
    Инициализация:
"""
R = np.zeros((N, N_t, 3))  # радиус-вектор (реальный) частиц
V = np.zeros((N, N_t, 3))  # скорость частиц
R[:, 0, :] = R0[:, 0, :]
V[:, 0, :] = V0[:, 0, :]
R_ = R.copy()              # радиус-вектор (расчетный/вспомогательный) частиц

K = np.zeros((N, N_t))     # кинетическая энергия частиц
P = np.zeros((N, N_t))     # потенциальная энергия частиц

"""
    Расчетные функции:
"""
def calcKinetic(p_i: int, t_i: int) -> float:
    ''' Расчет кинетической энергии частицы в данный момент времени
    Args:
        p_i (int): индекс частицы 
        t_i (int): индекс момента времени эксперимента
    Returns: 
        float: кинетическая энергия
    '''
    return M * np.linalg.norm(V[p_i, t_i])**2 / 2

def calcPotential(p_i: int, t_i: int) -> float:
    ''' Расчет потенциальной энергии частицы в данный момент времени
    Args:
        p_i (int): индекс частицы 
        t_i (int): индекс момента времени эксперимента
    Returns: 
        float: потенциальная энергия
    '''
    # Радиус-вектор до каждой частицы
    Delta = R[p_i, t_i, :] - R[:, t_i, :] 
    Delta = np.reshape(Delta[Delta != 0], (N - 1, 3))
    # Расстояние между каждой частицей 
    distance = np.linalg.norm(Delta, axis = 1)
    # Разность потенциалов между каждой частицей (начальный потенциал равен нулю)
    potentials = np.array([ShieldPotentialZBL(d) for d in distance])
    return np.sum(potentials, axis = 0)

def calcForce(p_i: int, t_i: int) -> np.array:
    """ Расчет результирующей силы, действующей на частицу в данный момент времени
    Arguments:
        p_i (int): индекс частицы 
        t_i (int): индекс момента времени эксперимента
    Returns: 
        np.array: результирующая сила в момент времени t_i
    """
    # Радиус-вектор до каждой частицы
    Delta = R[p_i, t_i, :] - R[:, t_i, :] 
    Delta = np.reshape(Delta[Delta != 0], (N - 1, 3))
    # Расстояние между каждой частицей 
    distance = np.linalg.norm(Delta, axis = 1)
    return W * np.sum(Delta * np.reshape(1 / distance**3, (N - 1, 1)), axis = 0)

def calcVelocity(p_i: int, t_i: int, prev_pos: Optional[np.array] = None) -> np.array:
    ''' Расчет скорости частицы в данный момент времени (по схеме Верле)
    Args:
        p_i (int): индекс частицы 
        t_i (int): индекс момента времени эксперимента
    Returns: 
        np.array: скорость частицы в момент времени t_i
    '''
    return  V[p_i, t_i, :] + H / (2 * M) * (calcForce(p_i, t_i + 1) - calcForce(p_i, t_i))

def calcPosition(p_i: int, t_i: int) -> np.array: 
    ''' Расчет последующего положения частицы (по схеме Верле)
    Args:
        p_i (int): индекс частицы 
        t_i (int): индекс момента времени эксперимента
    Returns: 
        np.array: положение частцы в момент времени t_i + 1
    '''
    return R[p_i, t_i, :] + H * V[p_i, t_i, :] + H**2 / (2 * M) * calcForce(p_i, t_i)

def сalcParticle(p_i: int, t_i: int) -> None:
    ''' Расчет состояния частицы
    Args:
        p_i (int): индекс частицы 
        t_i (int): индекс момента времени эксперимента
    Returns: 
        None
    '''
    K[p_i, t_i] = calcKinetic(p_i, t_i)
    P[p_i, t_i] = np.sum(calcForce(p_i, t_i)) / W
    
    R[p_i, t_i + 1, :] = calcPosition(p_i, t_i) 
    V[p_i, t_i + 1, :] = calcVelocity(p_i, t_i)

    # Детектирования прохождения частицы через стенку 
    components = np.where(abs(R[p_i, t_i + 1]) >= L)[0]
    for x_i in components:
        R[p_i, t_i + 1, x_i] -= np.sign(R[p_i, t_i + 1, x_i]) * (2 * L)
    
    
def calcExperiment() -> None:
    """ Моделироваение движения частиц """
    
    bar = IncrementalBar('Моделироварие', min = 2, max = N_t - 2, suffix = '%(percent).1f%% (%(eta)ds)')        
    for t_i in range(1, N_t - 1):
        for p_i in range(N):
            сalcParticle(p_i, t_i)
        bar.next()
    bar.finish()
    

def animate(n: int) -> None:
    """ Анимировение движения частиц
    Args:
        n (int): момент времени
    Returns: 
        None
    """
    # Отрисовка 3D сцены
    perspective.clear()
    perspective.set_xlim([-1.2 * L, 1.2 * L])
    perspective.set_ylim([-1.2 * L, 1.2 * L])
    perspective.set_zlim([-1.2 * L, 1.2 * L])
    perspective.set_xlabel(r"$X$")
    perspective.set_ylabel(r"$Y$")
    perspective.set_zlabel(r"$Z$")
    # Отрисовка 2D сцены в проекции на xy
    projection_xy.clear()
    projection_xy.set_xlim([-L, L])
    projection_xy.set_ylim([-L, L])
    projection_xy.set_xlabel(r"$X$")
    projection_xy.set_ylabel(r"$Y$")
    # Отрисовка 2D сцены в проекции на xz
    projection_xz.clear()
    projection_xz.set_xlim([-L, L])
    projection_xz.set_ylim([-L, L])
    projection_xz.set_xlabel(r"$X$")
    projection_xz.set_ylabel(r"$Z$")
    # Отрисовка 2D сцены в проекции на yz
    projection_yz.clear()
    projection_yz.set_xlim([-L, L])
    projection_yz.set_ylim([-L, L])
    projection_yz.set_xlabel(r"$Y$")
    projection_yz.set_ylabel(r"$Z$")
    
    """ Настройка отрисовывемой траектории
    Для того, чтобы не загромождать сцену траекториями частиц
    будут отрисовываться последние tail_lenght предыдущие
    местоположения частицы.
    """
    tail_end = 0        # конец "хвоста" чатсицы
    tail_lenght = 1000  # длина "хвоста частцы"
    if n >= tail_lenght:
        tail_end = n - tail_lenght
        
    # Анимирование
    lines = []
    for i in range(N):
        lines.append(perspective.scatter(R[i, n, 0], R[i, n, 1], R[i, n, 2]))
        lines.append(perspective.plot(R[i, tail_end:n, 0], R[i, tail_end:n, 1], R[i, tail_end:n, 2], linestyle = "solid", alpha = 0.5))
        
        lines.append(projection_xy.scatter(R[i, n, 0], R[i, n, 1]))
        lines.append(projection_xy.plot(R[i, tail_end:n, 0], R[i, tail_end:n, 1], linestyle = "solid", alpha = 0.5))
        
        lines.append(projection_xz.scatter(R[i, n, 0], R[i, n, 2]))
        lines.append(projection_xz.plot(R[i, tail_end:n, 0], R[i, tail_end:n, 2], linestyle = "solid", alpha = 0.5))
        
        lines.append(projection_yz.scatter(R[i, n, 1], R[i, n, 2]))
        lines.append(projection_yz.plot(R[i, tail_end:n, 1], R[i, tail_end:n, 2], linestyle = "solid", alpha = 0.5))
        
        if tail_end != 0: tail_end += 1
    return lines

if __name__ == "__main__":
    # Расчет первой точки по времени
    K[:, 0] = M * np.linalg.norm(V[:, 0, :], axis = 1)**2 / 2
    for p_i in range(N):
        P[p_i, 0]  = np.sum(calcForce(p_i, 0)) / W
    
    # Расчет второй точки по времени
    for p_i in range(N):
        R[p_i, 1, :] = R[p_i, 0, :] + H * V[p_i, 0, :]   
        V[p_i, 1, :] = V[p_i, 0, :] + H * calcForce(p_i, 1)

    # Расчет последующих моментов времени
    calcExperiment()
    K[:, -1] = M * np.linalg.norm(V[:, -1, :], axis = 1)**2 / 2

    # Анимириоваие траекторий 
    fig = plt.figure(layout = 'constrained', figsize = (18, 18))
    subfigs = fig.subfigures(1, 2, width_ratios = [3, 1])
    
    perspective = subfigs[0].subplots(1, 1, subplot_kw = {"projection": "3d"})  # сцена в 3D
    (projection_xy, projection_xz, projection_yz) = subfigs[1].subplots(3, 1, height_ratios =[1, 1, 1])  # сцены в 2D (проекции)
    
    anim = animation.FuncAnimation(fig, animate, frames = N_t, interval = 0.0001)
    
    # Построение траекторий частиц (при необходимости раскомментировать)
    # fig, ax = plt.subplots(1, 1, subplot_kwb = {"projection": "3d"})
    # ax.set_xlim([-1.2 * L, 1.2 * L])
    # ax.set_ylim([-1.2 * L, 1.2 * L])
    # ax.set_zlim([-1.2 * L, 1.2 * L])
    # ax.set_xlabel(r"$X$")
    # ax.set_ylabel(r"$Y$")
    # ax.set_zlabel(r"$Z$")
    # for i in range(N):
    #     ax.plot(R[i, :, 0], R[i, :, 1], R[i, :, 2])

    # График зависимости безрамерной суммарной энергии от времени 
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r"$t \cdot 10^{-15}$, c")
    ax.set_ylabel(r"$W_\sum$")
    
    for i in range(N):
       plt.plot(np.linspace(1, N_t, K[i, 0:-2].size), P[i, 0:-2] + K[i, 0:-2], alpha = 0.3, linestyle = "dashed", label = f"Суммарная энергия частицы №{i+1}") 
    plt.plot(np.linspace(1, N_t, K[i, 0:-2].size), np.sum(P[:, 0:-2] + K[:, 0:-2], axis = 0) / N, alpha = 1, linestyle = "solid", label = "Суммарная энергия системы")
    plt.legend()
    
    plt.grid(True, which = "major", alpha = 0.3)
    plt.grid(True, which = "minor", alpha = 0.1)
    plt.minorticks_on()
    plt.savefig("energies.png", dpi = 500)
    
    # График потенциальной энергии
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r"$t \cdot 10^{-15}$, c")
    ax.set_ylabel(r"$W_P$")
    
    for i in range(N):
       plt.plot(np.linspace(1, N_t, K[i, 0:-2].size), P[i, 0:-2], alpha = 0.3, linestyle = "dashed", label = f"Потенициальная энергия частицы №{i+1}") 
    plt.plot(np.linspace(1, N_t, K[i, 0:-2].size), np.sum(P[:, 0:-2], axis = 0) / N, alpha = 1, linestyle = "solid", label = "Потенициальная энергия системы")
    plt.legend()
    
    plt.grid(True, which = "major", alpha = 0.3)
    plt.grid(True, which = "minor", alpha = 0.1)
    plt.minorticks_on()
    plt.savefig("W_p.png", dpi = 500)
    
    # График кинетической энергии
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel(r"$t \cdot 10^{-15}$, c")
    ax.set_ylabel(r"$W_K$")
    
    for i in range(N):
       plt.plot(np.linspace(1, N_t, K[i, 0:-2].size), K[i, 0:-2], alpha = 0.3, linestyle = "dashed", label = f"Кинетическая энергия частицы №{i+1}") 
    plt.plot(np.linspace(1, N_t, K[i, 0:-2].size), np.sum(K[:, 0:-2], axis = 0) / N, alpha = 1, linestyle = "solid", label = "Кинетическая энергия системы")
    plt.legend()
    
    plt.grid(True, which = "major", alpha = 0.3)
    plt.grid(True, which = "minor", alpha = 0.1)
    plt.minorticks_on()
    plt.savefig("W_k.png", dpi = 500)

    plt.show()
    