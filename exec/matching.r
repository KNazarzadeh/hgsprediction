
set.seed(1234)
df.patients <- r_data_frame(n = 250, 
                            age(x = 30:78, 
                                name = 'Age'), 
                            sex(x = c("Male", "Female"), 
                                prob = c(0.70, 0.30), 
                                name = "Sex"))
df.patients$Sample <- as.factor('Patients')

