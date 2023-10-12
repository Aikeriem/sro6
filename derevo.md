# Импортируем необходимые библиотеки
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Загрузим набор данных о диабете
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Разделим данные на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируем и обучим дерево решений
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train, y_train)

# Сделаем прогнозы на тестовом наборе данных
y_pred = tree_classifier.predict(X_test)

# Оценим точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy}")

