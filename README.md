# aircraft-taxiing-vibrations
 
## Использование стиля ```'russian-font'``` в пакете ```SciencePlots```
1. Установить TexLive и TexStudio для всех пользователей.
2. Используя менеджер TexLive, установить все пакеты и коллекции с кириллическими шрифтами.
3. Удостовериться, что в переменную PATH **как для пользователя, так и для системы** прописаны пути *C:\texlive\2023\bin\windows*, *C:\Program Files\texstudio\texstudio.exe*, *C:\Users\'USER_NAME'\\.texlive2023*.
4. В случае появления ```FileNotFoundError: Matplotlib's TeX implementation searched for a file named 'lati0900.tfm' in your texmf tree, but could not find it```, выполнить команду **mktextmf lati0900**. Проверить, что сгенерированная метрика шрифта сохраняется в директории *C:\Users\'USER_NAME'\\.texlive2023*, в противном случае указать эту директорию в переменной PATH.
5. Выполнить п.4 для недостающих метрик шрифтов.
