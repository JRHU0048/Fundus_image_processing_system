// import request from '@/utils/request'

// export function getList(params) {
//   return request({
//     url: '/vue-admin-template/table/list',
//     method: 'get',
//     params
//   })
// }

// 从 '@/utils/request' 模块导入 request 函数，这个函数通常用于发起 HTTP 请求
// 一般来说，该函数可能对 axios 等请求库进行了封装，以满足项目特定的请求需求
import request from '@/utils/request'

/**
 * 该函数用于获取表格列表数据
 * @param {Object} params - 包含请求参数的对象，这些参数会被附加到请求的 URL 上作为查询参数
 * @returns {Promise} - 返回一个 Promise 对象，当请求成功时，会解析为服务器返回的响应数据；请求失败时，会抛出错误
 */
export function getList(params) {
    // 调用 request 函数发起 HTTP 请求
    return request({
        // 请求的 URL，这里是 '/vue-admin-template/table/list'
        url: '/vue-admin-template/table/list',
        // 请求的方法为 GET，用于从服务器获取资源
        method: 'get',
        // 将传入的 params 对象作为查询参数附加到请求的 URL 上
        params
    })
}