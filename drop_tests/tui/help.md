# *VirtualDropTest v0.1* Программа для моделирования копровых испытаний телескопических опор шасси

## Назначение

Данная программа предназначена для анализа динамики телескопических опор шасси самолета с одно- или двухкамерным амортизатором в условиях, воспроизводящих натурные копровые испытания или (в первом приближении) посадку самолета.

## Как пользоваться

Окно программы разделено на три вкладки:

1. Подготовка
2. Результаты
3. Сообщения

Во вкладке *Подготовка* выполняется загрузка файлов параметров модели амортизатора и пневматической шины. Файлы должны иметь расширение .json и структуру, рассмотренную в п. *Названия параметров - ключей .json-файлов*. В группе окон ввода *Параметры виртуальных испытаний* пользователь вводит параметры, характеризующие условия натурных копровых испытаний, подлежащих воспроизведению. Наконец, в группе *Анализ* пользователь может просмотреть список загруженных параметров при помощи кнопки *Проверить* и запустить виртуальные испытания нажатием кнопки *Анализ*. 

>*ВНИМАНИЕ!* Просмотр загруженных параметров возможен только после успешной загрузки .json-файлов! В противном случае пользователь при нажатии кнопки *Проверить* увидит всплывающее окно, предупреждающее, что данные для расчета не подготовлены. Пользователю следует закрыть всплывающее окно и повторить загрузку файлов, убедившись в правильности их пути и формата.

Во вкладке *Результаты* в случае успешного завершения расчета в таблице появится распечатка вектора состояния и силовых факторов по шагам интегрирования. В группе *Работа с файлами результатов* полученные векторы можно сохранить в текстовый файл. Помимо просмотра и сохранения результатов текущих виртуальных испытаний существует возможность загрузки и просмотра текстовых файлов предыдущих испытаний.

Во вкладке *Сообщения* в режиме реального времени выводится распечатка программных сообщений, свидетельствующих о начале, конце и результате выполнения каждой операции, инициированной пользователем.

## Уравнения математической модели

Общее уравнение динамики системы "опора - копер" имеет вид:
```
    2          
   d           
M⋅───(q̂(t)) = f̂,
    2          
  dt 
```
где ```M``` - матрица масс системы (```m``` - масса груза, установленного в клети копра, ```m₁``` - масса подвижных частей стойки, приведенная к оси колес, ```J₁``` - осевой момент инерции колес относительно оси их вращения, ```α``` - угол установки стойки):         
```
    ⎡ m + m₁    m₁⋅cos(α)  m₁⋅sin(α)  0 ⎤
    ⎢                                   ⎥
    ⎢m₁⋅cos(α)     m₁          0      0 ⎥
M = ⎢                                   ⎥,
    ⎢m₁⋅sin(α)      0         m₁      0 ⎥
    ⎢                                   ⎥
    ⎣    0          0          0      J₁⎦
```
```q̂(t)``` - вектор обобщенных координат (```y(t)``` - положение клети копра, ```u(t)``` - обжатие амортизатора, ```v(t)``` - прогиб штока, вычисленный по оси колес, ```φ(t)``` - угол поворота колес):

```
        ⎡y(t)⎤
        ⎢    ⎥
        ⎢u(t)⎥
q̂(t) = ⎢    ⎥,
        ⎢v(t)⎥
        ⎢    ⎥
        ⎣φ(t)⎦
```
а ```f̂``` - вектор обобщенных сил:
```
     ⎡            Fₐ + Fₜ - m⋅g - m₁⋅g          ⎤
     ⎢                                          ⎥
     ⎢-Fₛ - Fₓ⋅sin(α) + Fₜ⋅cos(α) - m₁⋅g⋅cos(α) ⎥
f̂ = ⎢                                          ⎥.
     ⎢-Fₚ + Fₓ⋅cos(α) + Fₜ⋅sin(α) - m₁⋅g⋅sin(α) ⎥
     ⎢                                          ⎥
     ⎣                    Mᵩ                    ⎦
```
Здесь:

* ```Fₐ``` - сила разгрузки, имитирующая подъемную силу крыла самолета;
* ```Fₛ``` - сила сопротивления изменения объему газа;
* ```Fₜ``` - вертикальная реакция, возникающая при обжатии пневматических шин;
* ```Fₓ``` - горизонтальная реакция, возникающая при качении шин с частичным проскальзыванием;
* ```Fₚ``` - сила трения в буксах, центрирующих шток амортизатора в стакане стойки;
* ```Mᵩ``` - тормозной момент, возникающий на оси колес стойки.

## Названия параметров - ключей .json-файлов

---
Разработано с помощью фреймворка ```Textual```. © 2023 ПАО <<Яковлев>> 