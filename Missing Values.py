from sklearn.impute import SimpleImputer
import missingno as msno
class MissingValues:
    def __init__(self,db):
        self.db=db
        self.table=db.isna().sum()
    
    def show_table(self,n=5):   
        miss_val_per=(self.table/len(self.db))*100
        miss_table=pd.concat([self.table,miss_val_per],axis=1)
        self.miss_table=miss_table.rename(columns={0:"Missing Values",1:"Percentage missing"})
        self.miss_table_sort=self.miss_table[self.miss_table.iloc[:,:]!=0].sort_values("Percentage missing",ascending=False).round(1)
        return self.miss_table_sort.head(n)
    
    def show_bar_graph(self):
        return msno.bar(self.db)
        
    def show_matrix(self):
        return msno.matrix(self.db)
        
    def heatmap(self):
        return msno.heatmap(self.db)
    
    def impute(self,strategy_numerical='mean',strategy_categorical='most_frequent',constant_numerical=0,constant_categorical="Unknown"):
        numerical_imputer=SimpleImputer(strategy=strategy_numerical,fill_value=constant_numerical)
        categorical_imputer=SimpleImputer(strategy=strategy_categorical,fill_value=constant_categorical)
        for col in self.db.columns:
            if self.db[col].dtype!='O' and self.db[col].isna().any():
                self.db[col]=numerical_imputer.fit_transform(self.db[[col]])
            elif self.db[col].isna().any():
                self.db[col]=categorical_imputer.fit_transform(self.db[[col]])
        return self.db
