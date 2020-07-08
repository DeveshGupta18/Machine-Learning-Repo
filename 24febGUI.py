
################################################
#                   TKINTER
###############################################

from tkinter import *
window = Tk()#for window GCI
window.title("IRIS APPLICAITON")    #title
window.geometry("300x400")
window.configure(background='cyan') #background color

#def function for submit button
    
def sub():
    subwindow=Tk()
    subwindow.title("Sub Window - Prediction")
    subwindow.geometry("400x400")
    
    Button (subwindow, text="exit",
        bg="cyan", 
        fg="black",
        height=2, 
        width=10,
        command=subwindow.destroy).place(x=95, y=300)

    
    subwindow.mainloop()

Label(window,
      text="IRIS",
      bg="white",
      fg="red",  
      font=("Times", "24", "bold italic"),
      relief="solid").pack()

Label(window,
      text="Petal Length",
      bg="white",
      fg="red",  
      font=("Times", "24", "bold italic"),
      relief="solid").place(x=20, y=20)

Label(window,
      text="Petal Width",
      bg="white",
      fg="red",  
      font=("Times", "24", "bold italic"),
      relief="solid").place(x=20, y=50)

Label(window,
      text="Sepal Length",
      bg="white",
      fg="red",  
      font=("Times", "24", "bold italic"),
      relief="solid").place(x=20, y=80)

Label(window,
      text="Sepal Width",
      bg="white",
      fg="red",  
      font=("Times", "24", "bold italic"),
      relief="solid").place(x=20, y=110)

Button (window, text="submit",
        bg="yellow", 
        fg="black",
        height=2, 
        width=10,
        command=sub).place(x=75, y=300)


Button (window, text="exit",
        bg="yellow", 
        fg="black",
        height=2, 
        width=10,
        command=window.destroy).place(x=175, y=300)

#ENTRY WIDGET
a=StringVar()
Entry(window, textvariable=a).place(x=30, y=50)
window.mainloop()

####################################################################################
####################################################################################

from tkinter import*
window=Tk()
window.geometry("400x600")
window.configure(background="light green")
window.title("IRIS CLASSIFIER")
photo=PhotoImage(file =r"C:\Users\DELL\Desktop\ML Files\myimage gui.png")
window.iconphoto(False, photo)

from PIL import Image, ImageTk
i=Image.open(r"C:\Users\DELL\Desktop\ML Files\download.jpg")
photo = ImageTk.PhotoImage(i)
label = Label(window,image=photo,height=100,width=100)
label.image = photo # keep a reference!
label.place(x=140,y=100)



Label(window,text="PLANT CLASSIFIER APP",
      font=("Helvetica",15,"bold"),relief="solid",
      bg="light green",fg="black").place(x=80,y=50)

#############entry label
Label(window,text="Sepal Length",bg="light green",
      fg="black",font=10,
      relief="solid").place(x=60,y=250)

Label(window,text="Sepal Width",bg="light green",
      fg="black",font=10,
      relief="solid").place(x=60,y=290)
Label(window,text="Petal Length",bg="light green",
      fg="black",font=10,
      relief="solid").place(x=60,y=330)
Label(window,text="Petal Width",bg="light green",
      fg="black",font=10,
      relief="solid").place(x=60,y=370)
######################### Label Entry widget
sl=StringVar()
sw=StringVar()
pl=StringVar()
pw=StringVar()

Entry(window,textvariable=sl).place(x=200,y=250)
Entry(window,textvariable=sw).place(x=200,y=290)
Entry(window,textvariable=pl).place(x=200,y=330)
Entry(window,textvariable=pw).place(x=200,y=370)
########gui deploy model####
def ml():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    data=load_iris()
    x=data.data
    y=data.target
    
    from sklearn.neighbors import KNeighborsClassifier
    model=KNeighborsClassifier()
    model.fit(x,y)
    x_test=[float(sl.get()),float(sw.get()),
            float(pl.get()),float(pw.get())]
    y_pred=model.predict([x_test,])
   # print("prediction is :",data.target_names[y_pred])
    if y_pred in data.target:
        if(data.target_names[y_pred]=="setosa"):
            win1=Tk()
            win1.geometry("150x150")
            Label(win1,
            text="Prediction Specie is Setosa",
            fg="Black",relief="sunken").pack()
            #i_w1=Image.open(r"C:\Users\chetna\Desktop\download.jpg")
            #photo_win1 = ImageTk.PhotoImage(i_w1)
            #label = Label(win1,image=photo_win1
            #              ,height=100,width=100)
            #label.image = photo_win1 # keep a reference!
            #label.pack()
            win1.mainloop
        elif(data.target_names[y_pred]=="versicolor"):
            win2=Tk()
            Label(win2,
            text="Prediction Specie is Versicolor",
            fg="Black",relief="sunken").pack()
            win2.mainloop
        else:
            win3=Tk()
            Label(win3,
            text="Prediction Specie is virginica",
            fg="Black",relief="sunken").pack()
            win3.mainloop

Button(window,text="Prediction",fg="black",
       bg="yellow",command=ml).place(x=100,y=450)
Button(window,text="Exit",fg="black",
       bg="yellow",width=10,
       command=window.destroy).place(x=200,y=450)
window.mainloop()
##############
import cv2
img=cv2.imread("path")
print(img)





