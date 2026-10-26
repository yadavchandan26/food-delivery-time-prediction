from prediction.data_processing import final_processing
from prediction.model_training import train_model
from prediction.evaluate_model import model_score,submission

data="\data\Food_Delivery_Times.csv"

def main():
    x_train,x_test,y_train,y_test=final_processing(data)

    model=train_model(x_train,y_train)
    model_score(model,x_train,y_train)
    y_pred=model.predict(x_test)
    submission(x_test,y_test,y_pred,output_path="output.csv")

if __name__=='__main__':
    main()