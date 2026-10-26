from sklearn.linear_model import LinearRegression

def train_model(x_train_scale,y_train):
    model=LinearRegression()
    model.fit(x_train_scale,y_train)

    return model
