from sklearn.linear_model import LinearRegression

def train_model(x_train_scale,x_test_scale,y_train,y_test):
    model=LinearRegression()
    model.fit(x_train_scale,y_train)

    training_score=model.score(x_train_scale,x_test_scale)
    print("training score is :",training_score)

    return model