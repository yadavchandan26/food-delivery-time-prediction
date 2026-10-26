import pandas as pd

def model_score(model,x_train_scale,y_train):
    score=model.score(x_train_scale,y_train)
    print("model score is:",score)

    return score

def submission(y_pred,output_path='submission.csv'):
    submission=pd.DataFrame({
        'Delivery Time':y_pred
    })

    submission.to_csv(output_path,index=True)
    print(f"submission completed and saved to {output_path}")
    return submission