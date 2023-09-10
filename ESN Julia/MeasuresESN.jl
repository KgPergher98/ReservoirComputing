#=
    MEDIDAS A SEREM UTILIZADAS PARA REDES ESN

    CRIADO POR KEVIN PERGHER
    16/07/2022
=#

function mean_squared_error(;data, pred_data)
    mse = sum((data .- pred_data) .^ 2)/size(data)[1]
    return mse
end

function normalized_mean_squared_error(;data, pred_data)
    nmse = sum((data .- pred_data) .^ 2)/size(data)[1]
    nmse = nmse/sum(data .^ 2)
    return nmse
end

function mean_absolute_percentage_error(;data, pred_data)
    mape = sum(abs.((data .- pred_data) ./ data))*(100 / size(data)[1])
    return mape
end