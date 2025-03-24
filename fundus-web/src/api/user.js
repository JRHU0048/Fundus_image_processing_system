// import request from '@/utils/request'

// export function login(data) {
//   return request({
//     url: '/vue-admin-template/user/login',
//     method: 'post',
//     data
//   })
// }

// export function getInfo(token) {
//   return request({
//     url: '/vue-admin-template/user/info',
//     method: 'get',
//     params: { token }
//   })
// }

// export function logout() {
//   return request({
//     url: '/vue-admin-template/user/logout',
//     method: 'post'
//   })
// }


// 从 '@/utils/request' 模块导入封装好的请求函数
// 该函数通常对底层的 HTTP 请求库（如 axios）进行了封装，方便在项目中统一处理请求配置和响应处理
import request from '@/utils/request';

/**
 * 发起用户登录请求
 * @param {Object} data - 包含登录所需信息的对象，通常包含用户名和密码等
 * @returns {Promise} - 返回一个 Promise 对象，代表请求的异步操作
 * 当请求成功时，Promise 会 resolve 服务器返回的响应数据；请求失败时，会 reject 错误信息
 */
export function login(data) {
    // 调用封装好的 request 函数发起登录请求
    return request({
        // 请求的 URL，指向登录接口
        url: '/vue-admin-template/user/login',
        // 请求方法为 POST，通常用于向服务器提交数据
        method: 'post',
        // 将传入的登录数据作为请求体发送到服务器
        data
    });
}

/**
 * 根据 token 获取用户信息
 * @param {string} token - 用户的身份验证令牌，用于标识用户身份
 * @returns {Promise} - 返回一个 Promise 对象，代表请求的异步操作
 * 当请求成功时，Promise 会 resolve 服务器返回的用户信息；请求失败时，会 reject 错误信息
 */
export function getInfo(token) {
    // 调用封装好的 request 函数发起获取用户信息的请求
    return request({
        // 请求的 URL，指向获取用户信息的接口
        url: '/vue-admin-template/user/info',
        // 请求方法为 GET，用于从服务器获取资源
        method: 'get',
        // 将 token 作为查询参数添加到请求 URL 中
        params: { token }
    });
}

/**
 * 发起用户登出请求
 * @returns {Promise} - 返回一个 Promise 对象，代表请求的异步操作
 * 当请求成功时，Promise 会 resolve 服务器返回的响应数据；请求失败时，会 reject 错误信息
 */
export function logout() {
    // 调用封装好的 request 函数发起登出请求
    return request({
        // 请求的 URL，指向登出接口
        url: '/vue-admin-template/user/logout',
        // 请求方法为 POST，通常用于告知服务器执行某个操作
        method: 'post'
    });
}

// 这段代码封装了三个与用户认证相关的请求函数，分别用于登录、获取用户信息和登出操作。
// 通过这些函数，项目可以方便地与服务器进行交互，处理用户的认证流程。