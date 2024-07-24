#我保留了一部分GPT-4o的注释，这样就知道我是学计科的
import numpy as np
import scipy.optimize as opt
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

def log_barrier_function(allocation_matrix, utility_matrix):
    """
    计算对数障碍函数的值

    参数:
    allocation_matrix (np.array): 物品分配矩阵
    utility_matrix (np.array): 效用矩阵

    返回值:
    float: 对数障碍函数值
    """
    num_players = utility_matrix.shape[0]
    total = 0
    for i in range(num_players):
        for j in range(num_players):
            if utility_matrix[i, j] > 0:
                total += (np.log(allocation_matrix[i, j]) +
                          np.log(np.sum(utility_matrix[i, :] * allocation_matrix[i, :])) +
                          np.log(np.log(np.sum(utility_matrix[i, :] * allocation_matrix[i, :])) - np.log(
                              utility_matrix[i, j])))
    return total

def fisher_objective_function(allocation_matrix, utility_matrix, barrier_weight):
    """
    计算 Fisher 模型的目标函数值

    参数:
    allocation_matrix (np.array): 物品分配矩阵
    utility_matrix (np.array): 效用矩阵
    barrier_weight (float): 负对数障碍函数的权重

    返回值:
    float: 目标函数值
    """
    barrier = -barrier_weight * log_barrier_function(allocation_matrix, utility_matrix)
    return np.sum(allocation_matrix) + barrier

def arrow_debreu_objective_function(allocation_matrix, utility_matrix, barrier_weight):
    """
    计算 Arrow-Debreu 模型的目标函数值

    参数:
    allocation_matrix (np.array): 物品分配矩阵
    utility_matrix (np.array): 效用矩阵
    barrier_weight (float): 负对数障碍函数的权重

    返回值:
    float: 目标函数值
    """
    barrier = -barrier_weight * log_barrier_function(allocation_matrix, utility_matrix)
    return np.sum(allocation_matrix) + barrier

def interior_point_algorithm(utility_matrix, initial_allocation, barrier_weight, tolerance, max_iterations, objective_function):
    """
    使用内点法求解给定目标函数

    参数:
    utility_matrix (np.array): 效用矩阵
    initial_allocation (np.array): 初始物品分配矩阵
    barrier_weight (float): 负对数障碍函数的权重
    tolerance (float): 收敛公差
    max_iterations (int): 最大迭代次数
    objective_function (function): 目标函数

    返回值:
    np.array: 最优物品分配矩阵
    """
    allocation_matrix = initial_allocation
    num_players = utility_matrix.shape[0]

    for iteration in range(max_iterations):
        logging.info(f"Iteration {iteration + 1}")

        def objective_function_wrapper(flat_allocation):
            reshaped_allocation = flat_allocation.reshape((num_players, num_players))
            return objective_function(reshaped_allocation, utility_matrix, barrier_weight)

        flat_initial_allocation = allocation_matrix.flatten()
        result = opt.minimize(
            objective_function_wrapper, flat_initial_allocation, method='trust-constr',
            options={'verbose': 1},
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x.reshape((num_players, num_players)), axis=0) - 1}],
            bounds=[(0, None)] * num_players ** 2
        )

        allocation_matrix = result.x.reshape((num_players, num_players))
        logging.info(f"Updated allocation matrix:\n{allocation_matrix}")

        if result.success and np.linalg.norm(result.jac) < tolerance:
            logging.info("Converged!")
            break

    return allocation_matrix

def generate_large_utility_matrix(size, low=1, high=10):
    """
    生成指定大小的效用矩阵，效用值为整数

    参数:
    size (int): 矩阵的大小
    low (int): 随机整数的最小值（包含）
    high (int): 随机整数的最大值（不包含）

    返回值:
    np.array: 生成的效用矩阵
    """
    return np.random.randint(low, high, size=(size, size))

def main(model='fisher'):
    # 定义玩家数
    num_players = 10  # 可以改为更大的值以生成更大的效用矩阵

    # 生成效用矩阵
    utility_matrix = generate_large_utility_matrix(num_players)

    # 初始化物品价格或初始购买数量
    initial_allocation = np.ones((num_players, num_players)) / num_players

    logging.info(f"Initial utility matrix:\n{utility_matrix}")
    logging.info(f"Initial allocation matrix: \n{initial_allocation}")

    # 设定参数
    barrier_weight = 0.01
    tolerance = 1e-6
    max_iterations = 50

    if model == 'fisher':
        result = interior_point_algorithm(utility_matrix, initial_allocation, barrier_weight, tolerance, max_iterations, fisher_objective_function)
    elif model == 'arrow-debreu':
        result = interior_point_algorithm(utility_matrix, initial_allocation, barrier_weight, tolerance, max_iterations, arrow_debreu_objective_function)
    else:
        raise ValueError("Unknown model specified.")

    logging.info(f"Optimal allocation matrix for {model} model:\n{result}")

if __name__ == "__main__":
    main('fisher')  # 可以改为 'arrow-debreu' 选择模型