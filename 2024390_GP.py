#Nirvan 2024390
import speech_recognition as sp
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import webbrowser
import imdb
listener=sp.Recognizer()
en=pyttsx3.init()
en.setProperty('rate',120)
def spk(txt):
    try:
        en.say(txt)
        en.runAndWait()
    except:
        pass
def hukum():
    try:
        with sp.Microphone() as mic:
            print('please speak & speak after 2 second pause')
            voice=listener.listen(mic)
            command=listener.recognize_google(voice)
            command=command.lower()
            print(command)
    except:
        pass
    return command
def vaan():
    cmd=hukum()
    if 'play' in cmd:
        sng=cmd.replace('play','')
        spk('playing '+sng)
        pywhatkit.playonyt(sng)
    elif 'time' in cmd:
        t=datetime.datetime.now().strftime('%I:%M')
        s='current time is '+ t
        print(s)
        spk(s)
    elif 'open youtube' in cmd:
        spk('opening youtube master')
        webbrowser.open('www.youtube.com')
    elif 'open facebook' in cmd:
        spk('opening facebook master')
        webbrowser.open('www.facebook.com')
    elif 'open twitter' in cmd or 'open x' in cmd:
        spk('opening twitter')
        webbrowser.open('www.twitter.com')
    elif 'who is' in cmd or 'info about'in cmd or 'wiki' in cmd or 'information about' in cmd:
        try:
            if 'info about' in cmd:
                p=cmd.replace('info about','') 
            if 'information about' in cmd:
                p=cmd.replace('information about','')
            elif 'wiki' in cmd:
                p=cmd.replace('wiki','')
            else:
                p=cmd.replace('who is','')
        except:
            spk('No information found on wikipedia master')
        info=wikipedia.summary(p,3)
        print(info)
        spk(info)
    elif 'joke' in cmd:
        spk(pyjokes.get_joke())
    elif 'search' in cmd:
        t=cmd.replace('search','')
        spk('master searched on internet and opening for you')
        pywhatkit.search(t)
    elif 'movie' in cmd:
        t=str(cmd.replace('open',''))
        x=imdb.IMDb()
        spk('telling you about movie')
        y=x.search_movie(t)
        spk('Master I have provided you this in written')
        for i in range(len(y)):
            try:
                print(y[i])
            except:
                pass
    elif 'stop' in cmd:
        spk('break')
        return 'stop'
    else:
        spk('Please say the command again master')
print('say open first to open sites like youtube,facebook,twitter')
print('say search first to search something on google')
print('say time to ask time')
print('say play to play any song with name')
print('say who is/info about/information about to get wikipedia summary')
print('say joke to ask joke')
print('say movie then name to search for movie')
print('say stop to terminate')
en.say('Master I am your assistant Vaan')
en.say('Order me for help')
en.runAndWait()
while True:
    x=vaan()
    if x!='stop':
        continue
    else:
        break
def _test():
    assert spk('jai')==None
    assert spk('jai ho Thakur Nirvan Singh of Barsi Gharana 24th generation Thakur of Barsi to Jhunjhunu states')==None
_test()