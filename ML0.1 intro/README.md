### Входной контроль

#### Цель работы

Проверить знания базовых инструментов, которые используются для обращения с данными в проектах по машинному обучению - библиотек numpy, pandas, средств визуализации matplotlib, seaborn.

#### Методические указания

Для успешного выполнения заданий и проектов по машинному обучению необходимо постоянное манипулирование данными в определенных форматах. Очень облегчают эту задачу специализированные библиотеки. В экосистеме Python почти всегда используются модули numpy и pandas для анализа и преобразования данных. Поэтому для эффективной работы с моделями машинного обучения необходимо виртуозно владеть этими базовыми библиотеками.

Среди базовых возможностей языка программирования Python при работе с анализом данных и машинным обучением чаще всего используются следующие:

- индексирование массивов и срезы
- генераторные выражения с условиями и циклами
- анонимные функции 

Среди средств библиотеки numpy чаще всего применяются следующие инструменты:

- функции генерации массивов arange, linspace, logspace
- функция генерации сетки mgrid
- функции создания матриц diag, zeroes, ones
- создание случайных значений - пакет random
- форма массива - shape
- изменение формы массива - reshape
- оператор многоточия - ...
- сложное индексирование
- индексные маски
- поэлементные операции

Библиотека pandas особенна полезна для манипуляции двумерными массивами информации. К самым часто используемым ее возможностям в машинном обучении относятся следующие:

- сохранение и чтение файлов в формате csv, xls, xlsx
- аналитические функции info, describe, head
- итерация по строкам и столбцам, обращение к строкам и столбцам по номерам и индексам/названиям
- объединения и соединения таблиц
- агрегатные функции и группировка
- замена отдельных значений в таблице в том числе по условию

Кроме того нам часто придется визуализировать данные в виде различных графиков, гистограмм. Средства визуализации тоже очень важны и полезны. Аналитик должен владеть разными типами графиков и выбирать наиболее подходящий для решения конкретной задачи. Помните, что визуализация нужна для того, чтобы сделать видимыми какие-то скрытые зависимости в данных. То есть визуализация должна быть простой, наглядной и понятной, с явными целями и выводами.

Эту и все последующие работы следует выполнять в ноутбучном редакторе Python, который позволяет выполнять код по ячейкам. Такой формат работы больше подходит для прототипирования и построения проекта по анализу данных и машинному обучению. Рекомендуется использовать Jupyter notebook или Google Colab.

#### Задания для самостоятельного выполнения

1. С помощью массивов numpy создайте таблицу умножения.
1. Создайте функцию, которая принимает как аргументы целое число N и первый элемент (вещественное число), и разность (вещественное число) и создает матрицу numpy по диагонали, которой располагаются первые N членов арифметической прогрессии.
1. Сгенерируйте средствами numpy матрицу А 5 на 5, содержащую последовательные числа от 1 до 25. Используя срезы извлеките в плоский массив все нечетные элементы этой матрицы.
1. Создайте двумерный массив, содержащий единицы на границе и нули внутри.
1. Создайте две матрицы размером (5,5). Одна матрица содержит 5 в шахматном порядке как в задаче домашнего задания, другая имеет треугольную форму содержащую 5 на основной диагонали и в позициях выше ее, а ниже все 0. Посчитайте их детерминант и найдите обратные матрицы.
1. С помощью pandas загрузите датасет для предсказания цены квартиры, прилагающийся к этой работе.
1. Выведите на экран несколько первых и несколько последний строк файла.
1. Выведите с помощью методов pandas основные количественные параметры датасета: количество строк и столбцов, тип данных каждого поля, количество значений в каждом столбце, шкала измерения каждого численного поля.
1. Удалите из таблицы столбцы, содержащие идентификаторы, переименуйте все оставшиеся названия колонок на русском языке.
10. Выведите отдельно столбец, содержащий цену, по номеру и названию. Выведите первую, десятую и предпоследнюю строку таблицы по номеру и по индексу.
1. Выделите в отдельную таблицу последние десять строк. Уберите в ней столбец с ценой. Склейте ее с первоначальной таблицей при помощи append. Заполните отсутствующие значения цены средним по таблице.
1. Выделите пять последних колонок в отдельную таблицу. Удалите в ней строки, в которых цена ниже среднего. Присоедините эту таблицу к изначальной (выберите самый подходящий тип соединения).
1. Выведите таблицу, содержащую среднюю цену и количество квартир на каждом этаже из первоначального набора данных.
1. Сохраните получившуюся таблицу в файлы формата csv и xlsx. Прочитайте их и убедитесь, что данные отображаются корректно.
1. Создайте в Excel или другом табличном редакторе таблицу, содержащую несколько численных и текстовых полей. Прочитайте ее в программу при помощи pandas.
1. Дальнейшие задания производите используя изначальную версию датасета. Должны быть подписаны названия графиков, названия осей, указаны значения на осях. Оцениваться будет использование количества различных атрибутов при построении графиков и визуальная красота.
1. Постройте круговую диаграмму для признака Rooms, иллюстрирующую количество квартир в процентах в зависимости от количества комнат. Сделайте сектор с наибольшим числом квартир выдвинутым.
1. Постройте гистограмму по целевой переменной Price. Оцените визуально, по какой цене продаётся наибольшее количество квартир.
1. Постройте диаграммы рассеяния для признаков Rooms, Square, HouseFloor, HouseYear в зависимости от целевой переменной Price в одной области figure. Оцените визуально, есть ли среди них такие, на которых разброс точек близок к линейной функции.
20. Постройте ядерную оценку плотности целевой переменной Price. Оцените визуально, напоминает ли полученный график нормальное распределение. Постройте двумерную ядерную оценку плотности для целевой переменной Price и признака HouseFloor, затем оцените визуально на каких этажах и по какой цене продаётся основная масса квартир.
1. Постройте ящиковую диаграмму признака Square. Оцените визуально имеются ли выбросы, и, если да, то начиная с какого размера площади значение признака можно считать выбросом.
1. При помощи сетки графиков PairGrid визуализируйте попарные отношения признаков Rooms, Square, HouseFloor, HouseYear, Price следующим образом: на диагонали - гистограммы, под диагональю - ядерные оценки плотности, над диагональю - диаграммы рассеяния. По результатам визуализации сделайте выводы.
1. Постройте тепловую карту матрицы корреляции (df.corr()) признаков Rooms, Square, HouseFloor, HouseYear, Price. По ней определите, какие признаки являются зависимыми (у таких признаков коэффициент корреляции близок к единице).


#### Контрольные вопросы

1. Какие структуры данных используются в Numpy? В чем их отличие от списков Python?
1. Какие функции для генерации массивов использует Numpy?
1. Какие способы предлагает Numpy для извлечения данных из массивов?
1. Что такое векторизация кода и почему это ускоряет работу программ?
1. Какие виды матричных операций реализованы в Numpy?
1. Какие функции используются для преобразования формы, размера и соединения массивов?
1. Какие две главные структуры данных используются в pandas? В чем их отличие?
1. Как происходит объединение двух таблиц в pandas?
1. Зачем нужны и как работают индексы в pandas.
1. Построение каких основных видов графиков используется при анализе данных в машинном обучении?
1. В чём разница между библиотеками matplotlib и seaborn? Каковы преимущества каждой из них?
1. Как задать размер графика в matplotlib?
1. Как установить стили в seaborn?
1. Для чего используют подграфики subplots?
1. Какие основные типы графиков реализованы в matplotlib? Что изображается на ящиковой диаграмме?
1. Как поменять палитру цветов у тепловой карты?


#### Дополнительные задания

1. Изучите документацию модулей numpy, pandas, matplotlib по тем темам, которые упомянуты в методических указаниях.
1. Изучите приемы работы с датой и временем: модуль [strftime](http://strftime.org/).
1. Попробуйте загружать как локальный файл, содержащий данные, так и файл, размещенный на хостинге (например, GitHub) по URL.
1. Продемонстрируйте разные типы соединения таблиц в pandas.
1. Напишите SQL-аналоги всех операций, которые мы проводили в лабораторной с помощью pandas.
1. В ноутбуке
1. Постройте график jointplot (гибрид scatterplot и histogram) из библиотеки seaborn. Постройте график violinplot (гибрид boxplot и ядерной оценки плотности) из библиотеки seaborn.
1. Продемонстрируйте работу с сеткой подзаголовков FacetGrid.
1. Продемонстрируйте работу с трехмерными графиками. Создайте интерактивный график.

Типы данных каждого столбца:
Id                 int64
DistrictId         int64
Rooms            float64
Square           float64
LifeSquare       float64
KitchenSquare    float64
Floor              int64
HouseFloor       float64
HouseYear          int64
Ecology_1        float64
Ecology_2         object
Ecology_3         object
Social_1           int64
Social_2           int64
Social_3           int64
Healthcare_1     float64
Helthcare_2        int64
Shops_1            int64
Shops_2           object
Price            float64
dtype: object 