 
import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('rf_regressor.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(StoreType, Assortment,StateHoliday, Promo):   
 
    # Pre-processing user input    
    if StoreType == "a":
        StoreType = 0
    elif StoreType=='b':
        StoreType=1
    elif StoreType=='c':
        StoreType=2
    else:
        StoreType=3
 
    if Assortment == "a":
        Assortment = 0
    elif Assortment=='b':
        Assortment=1
    elif Assortment=='c':
        Assortment=2
    else:
        Assortment=3
        
    if StateHoliday == "0":
        StateHoliday = 0
    elif StateHoliday=='a':
        StateHoliday=1
    elif StateHoliday=='b':
        StateHoliday=2
    else:
        StateHoliday=3   
 
    if Promo==0:
        Promo = 0
    else:
        Promo= 1  
 

 
    # Making predictions 
    prediction = classifier.predict( 
        [[StoreType, Assortment,StateHoliday, Promo]])
     
    if prediction == 0:
        pred = 'Rejected'
    else:
        pred = 'Approved'
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Loan Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    StoreType = st.selectbox('StoreType',("a","b","c","d"))
    Assortment = st.selectbox('Assortment',("a","b","c","d")) 
    StateHoliday = st.selectbox('Assortment',("0","a","b","c")) 
    Promo=st.selectbox('Promo',("0","1")) 
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(StoreType, Assortment,StateHoliday, Promo) 
        st.success('Your loan is {}'.format(result))
        
     
if __name__=='__main__': 
    main()

 
    