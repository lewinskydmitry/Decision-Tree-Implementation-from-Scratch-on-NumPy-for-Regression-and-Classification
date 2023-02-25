import numpy as np
from sklearn.base import BaseEstimator
import bisect as bs



class DecisionTree(BaseEstimator):

    def __init__(self, max_depth = 10, min_samples_split = 2, criterion_name = 'еntropy'):
        # Максимальная глубина дерева
        self.max_depth = max_depth
        # Минимальное число семплов для деления
        self.min_samples_split = min_samples_split
        # Критерий деления
        self.criterion_name = criterion_name
        self.current_depth = 0


    # Функция для поиска лучшего разделения
    def __find_best_split(self, data, target):
        # Прирост информации
        max_gain = 0
        # Номер колонки
        col = None
        # Значение для разделения
        spl_val = None
        # Итерируемся по всем столбцам

        for i, c in enumerate(data.T):
            # Придаем данным нужную форму
            c = c.reshape(1,-1)[0]
            # Если все значения в столбце одинаковые - не рассматриваем этот столбец
            if np.all(c == c[0]):
                continue
            # Находим прирост информации при текущих параметрах
            gain, cur_spl_val = self.__find_best_split_for_attr(c, target)
            # Если отдает пустое значение - значит останавливаем цикл
            if cur_spl_val == None:
                break
            # Если нашли лучшее значение (или единственное), то принимаем его
            if gain > max_gain or spl_val == None:
                max_gain, col, spl_val = gain, i, cur_spl_val
        return col, spl_val, max_gain


    def __find_best_split_for_attr(self, col, target):
        max_gain = 0
        spl_val = None
        new_arr = np.array([col, target.reshape(1,-1)[0]])
        # sort the columns w.r.t. the first row values
        sarr = new_arr[:, new_arr[0].argsort()]
        # Чтобы итерироваться только по уникальным значениям
        indices = np.where(sarr[1, :-1] != sarr[1, 1:])[0]
        for threshold in indices:
            # Находим индекс разделения
            j = bs.bisect_right(sarr[0], sarr[0, threshold])
            if j > sarr.shape[1] - 1:
                continue
            median = (sarr[0, j - 1] + sarr[0, j]) / 2
            # Создаем левые и правые части
            y_left = sarr[1, :j]
            y_right = sarr[1, j:]
            y_left = y_left.reshape(-1,1)
            y_right = y_right.reshape(-1,1)
            # Считаем прирост от такого разделения
            gain = self.__inform_gain(target, y_left, y_right, self.criterion_name)
            # Если прирост больше чем был или если разделяющее число пустое, то переназначаем их
            if gain > max_gain or spl_val == None:
                max_gain, spl_val = gain, median
        return max_gain, spl_val

    
    def fit(self, data, target):
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        if self.criterion_name in ['gini','entropy']:
            self.n_classes = np.max(target) + 1
            self.__fit_classification(data, target, node={}, depth=0)
        else:
            self.__fit_regression(data, target, node={}, depth=0)

########################################################################
    def __fit_classification(self, data, target, node, depth):
        if node is None or len(target) == 0:
            return None
        if np.all(target == target[0]): # monocromatic
            # make a leaf
            dict, total = self.__counter(self.__one_hot_encode(self.n_classes, target))
            prob = np.array(list(dict.values())) / total
            node = {'class': target[0][0],
                    'samples': len(target),
                    'prob' : prob}
            for key in dict.keys():
                node[key] = dict[key]
            self.root = node
            return node
        else:
            # find best split based on information gaiz
            col, spl_val, gain = self.__find_best_split(data, target)
            dict, total = self.__counter(self.__one_hot_encode(self.n_classes, target))
            cl = int(max(dict, key = dict.get))
            
            if spl_val == None or depth == self.max_depth:
                # leaf
                prob = np.array(list(dict.values())) / total
                node = {'class': np.argmax(prob),
                        'samples': len(target),
                        'prob':prob}
                for key in dict.keys():
                    node[key] = dict[key]
                self.root = node
                return node
            
            dict, total = self.__counter(self.__one_hot_encode(self.n_classes, target))
            node = {'attr': col,
                    'index_col': col,
                    'split_value': spl_val,
                    'class': cl,
                    'samples': total}
            for key in dict.keys():
                    node[key] = dict[key]
            
            t_left = target[data[:, col] < spl_val]
            t_right = target[data[:, col] >= spl_val]
            node['left'] = self.__fit_classification(data[data[:, col] < spl_val], t_left, {}, depth + 1)
            node['right'] = self.__fit_classification(data[data[:, col] >= spl_val], t_right, {}, depth + 1)
            self.root = node
            return node

#################################################################

    def __fit_regression(self, data, target, node, depth):
        if node is None or len(target) == 0:
            return None
        if np.all(target == target[0]): # monocromatic
            # make a leaf
            if self.criterion_name == 'variance':
                node = {'samples': len(target),
                        'value' : np.mean(target)}
            else:
                node = {'samples': len(target),
                        'value' : np.median(target)}
            self.root = node
            return node
        else:
            # find best split based on information gain
            col, spl_val, gain = self.__find_best_split(data, target)
            
            if spl_val == None or depth == self.max_depth:
                # leaf
                if self.criterion_name == 'variance':
                    node = {'samples': len(target),
                            'value' : np.mean(target)}
                else:
                    node = {'samples': len(target),
                            'value' : np.median(target)}
                self.root = node
                return node
                
            if self.criterion_name == 'variance':
                node = {'attr': col,
                        'index_col': col,
                        'split_value': spl_val,
                        'value': np.mean(target),
                        'samples': len(target)}
            else:
                node = {'attr': col,
                        'index_col': col,
                        'split_value': spl_val,
                        'value': np.median(target),
                        'samples': len(target)}
            t_left = target[data[:, col] < spl_val]
            t_right = target[data[:, col] >= spl_val]
            node['left'] = self.__fit_regression(data[data[:, col] < spl_val], t_left, {}, depth + 1)
            node['right'] = self.__fit_regression(data[data[:, col] >= spl_val], t_right, {}, depth + 1)
            self.root = node
            return node

########################################################################

    def predict(self, data):
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        results = np.zeros(data.shape[0])
        if self.criterion_name in ['gini','entropy']:
            for i, row in enumerate(data):
                results[i] = self.__predict_classification(row)
        else:
            for i, row in enumerate(data):
                results[i] = self.__predict_regression(row)
        return results

    def predict_proba(self,data):
        if len(data.shape) == 1:
            data = data.reshape(-1,1)
        results = []
        if self.criterion_name in ['gini','entropy']:
            for i, row in enumerate(data):
                results.append(self.__predict_proba(row))
        else:
             raise Exception('Regression problem has not predict_proba!')
        return results


    def __predict_classification(self, row):
        node = self.root
        while 'attr' in node:
            if row[node['index_col']] < node['split_value']:
                node = node['left']
            else:
                node = node['right']
        return node['class']

    def __predict_regression(self, row):
        node = self.root
        while 'attr' in node:
            if row[node['index_col']] < node['split_value']:
                node = node['left']
            else:
                node = node['right']
        return node['value']
    
    def __predict_proba(self, row):
        node = self.root
        while 'attr' in node:
            if row[node['index_col']] < node['split_value']:
                node = node['left']
            else:
                node = node['right']
        return node['prob']

        
########## HELPER FUNCTION ##############################
    def __counter(self, data):
        dict = {}
        labels_sum = data.sum(axis = 0)
        for label in range(len(labels_sum)):
            dict[label] = labels_sum[label]
        return dict, np.sum(labels_sum)


    def __one_hot_encode(self, n_classes, y):
        y_one_hot = np.zeros((len(y), int(n_classes)), dtype=float)
        y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
        return y_one_hot


    def __entropy(self, y):
        n_classes = int(np.max(y)) + 1
        ohe = self.__one_hot_encode(n_classes, y)
        dict, total = self.__counter(ohe)
        entropy = 0.0
        for key in dict:
            p = dict[key] / total
            # entropy must be more than zero
            val = -p * np.log2(p + 0.0005) if -p * np.log2(p + 0.0005) > 0 else 0
            entropy += val
        return entropy


    def __gini(self, y):
        n_classes = int(np.max(y)) + 1
        ohe = self.__one_hot_encode(n_classes, y)
        dict, total = self.__counter(ohe)
        gini = 0.0
        for key in dict:
            gini += (dict[key] / total)**2
        return 1 - gini


    def __variance(self, y):
        return 1/len(y) * sum( (y - np.mean(y))**2 )

    def __mad_median(self, y):
        return 1/len(y) * sum( np.abs(y - np.mean(y)) )


    def __split_eval(self, y_left, y_right, metric):
        if metric in ['Entropy','Gini']:
            n_classes_l = np.max(y_left) + 1
            n_classes_r = np.max(y_right) + 1
            _,total_l = self.__counter(self.__one_hot_encode(n_classes_l, y_left))
            _,total_r = self.__counter(self.__one_hot_encode(n_classes_r, y_right))
            total = total_l + total_r
        else:
            total_l = len(y_left)
            total_r = len(y_right)
            total = total_l + total_r
            
        if metric == 'entropy':
            return total_l / total * self.__entropy(y_left) \
                + total_r / total * self.__entropy(y_right)

        if metric == 'gini':
            return total_l / total * self.__gini(y_left) \
                + total_r / total * self.__gini(y_right)
        
        if metric == 'variance':
            return total_l / total * self.__variance(y_left) \
                + total_r / total * self.__variance(y_right)

        if metric == 'mad_median':
            return total_l / total * self.__mad_median(y_left) \
                + total_r / total * self.__mad_median(y_right)


    def __inform_gain(self, y_subset, y_left, y_right, metric):
        if metric == 'entropy':
            return self.__entropy(y_subset) \
                - self.__split_eval(y_left, y_right,metric = metric)

        if metric == 'gini':
            return self.__gini(y_subset) \
                - self.__split_eval(y_left, y_right,metric = metric)

        if metric == 'variance':
            return self.__variance(y_subset) \
                - self.__split_eval(y_left, y_right,metric = metric)

        if metric == 'mad_median':
            return self.__mad_median(y_subset) \
                - self.__split_eval(y_left, y_right,metric = metric)