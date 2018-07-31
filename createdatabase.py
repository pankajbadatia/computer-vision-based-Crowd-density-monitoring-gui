# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 17:07:43 2017

@author: Pankaj
"""

import sqlite3

def createTable():
    connection = sqlite3.connect('rfid.db')

    connection.execute("CREATE TABLE USERS(name TEXT NOT NULL,mobilenumber TEXT,rfidnumber TEXT)")

    connection.execute("INSERT INTO USERS VALUES(?,?,?)",('pankaj','9021327935','740B73A9'))

    connection.commit()

    result = connection.execute("SELECT * FROM USERS")
    
    for data in result:
        print("Name : ",data[0])
        print("Mobile number : ",data[1])
        print("Rfid number :",data[2])

    connection.close()

createTable()