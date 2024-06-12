  Как преобразовать dataset Severstal в подходящий dataset для YOLO
Распаковываем датасет северстали в одну папку (рис. 1). 
 
![image](https://github.com/Nosikmov/Kaggle-Dataset-to-YOLO/assets/168178686/a426fd7d-f404-4d62-9e21-b1f30485c09f)

Исправляем столбцы в файле sample_submission.csv, так как нам требуется чтобы столбцы были вида «Название файла», «Номер класса», «Координаты». Первую строку в файле так же требуется удалить, заголовок нам не потребуется. Объединяем полученные файлы.
Все изображения перемещаем в одну папку. В .csv файле есть строка с файлом название которого «ba0c890a2.jpg» у этой строки в третьем столбце лишний символ. 
Строка 5728 должна иметь аннотацию к файлу, но без указания имени файла, строку нужно удалить.
Заходим в редактор кода и настраиваем пути до нужных папок.
![image](https://github.com/Nosikmov/Kaggle-Dataset-to-YOLO/assets/168178686/fa78a7ab-f9d8-4856-963c-78b9cd99f1bd)
Запускаем скрипт и получаем новую папку output и дожидаемся окончания выполнения скрипта с сообщением «Программа завершена».
В папке output должно появится две папки train и valid, а внутри labels и images, которые являются dataset’ом. 
