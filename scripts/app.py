 
import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('decision_tree.pkl', 'rb') 
decision_tree = pickle.load(pickle_in)
 
@st.cache()
# defining the function which will make the prediction using the data which the user inputs 
def prediction(StoreType ,Assortment,Date,IsHoliday,IsWeekend,IsPromo):
  #preprocessinh
  if StoreType == "a":
    StoreType = 0
  elif StoreType == "b":
    StoreType = 1
  elif StoreType == "c" :
    StoreType=2
  else:
    StoreType=3

  # Making predictions 
  prediction =decision_tree( 
        [[Sales]])
     
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
    <h1 style ="color:black;text-align:center;">Streamlit Sales Prediction ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    storetype = st.selectbox('StoreType',('a','b','c','d'))
    
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(StoreType ,Assortment,Date,IsHoliday,IsWeekend,IsPromo) 
        st.success('Your loan is {}'.format(result))
        print(LoanAmount)
     
if __name__=='__main__': 
    main()    
  

 
    