// 引入 Mock.js 库，用于生成模拟数据和拦截 Ajax 请求
const Mock = require('mockjs');
// 从 utils.js 文件中引入 param2Obj 函数，该函数用于将 URL 参数转换为对象
const { param2Obj } = require('./utils');

// 引入 user.js 文件中定义的模拟数据配置
const user = require('./user');
// 引入 table.js 文件中定义的模拟数据配置
const table = require('./table');

// 将 user 和 table 中的模拟数据配置合并到一个数组中
const mocks = [
  ...user,
  ...table
];

// 该函数用于在前端进行模拟请求，需要谨慎使用，因为它会重新定义 XMLHttpRequest，
// 这可能会导致许多第三方库失效（例如进度事件）
function mockXHR() {
  // 对 Mock.js 的 XHR 原型的 send 方法进行补丁处理
  // 参考：https://github.com/nuysoft/Mock/issues/300
  // 保存原始的 send 方法到 proxy_send 属性
  Mock.XHR.prototype.proxy_send = Mock.XHR.prototype.send;
  // 重写 send 方法
  Mock.XHR.prototype.send = function() {
    // 如果存在自定义的 xhr 对象
    if (this.custom.xhr) {
      // 将当前请求的 withCredentials 属性赋值给自定义 xhr 对象
      this.custom.xhr.withCredentials = this.withCredentials || false;

      // 如果当前请求设置了 responseType
      if (this.responseType) {
        // 将当前请求的 responseType 赋值给自定义 xhr 对象
        this.custom.xhr.responseType = this.responseType;
      }
    }
    // 调用原始的 send 方法并传递参数
    this.proxy_send(...arguments);
  };

  // 该函数用于将 Mock.js 的响应处理函数转换为类似 Express 请求处理函数的形式
  function XHR2ExpressReqWrap(respond) {
    return function(options) {
      let result = null;
      // 如果 respond 是一个函数
      if (respond instanceof Function) {
        // 从 options 中解构出请求体、请求方法和请求 URL
        const { body, type, url } = options;
        // 调用 respond 函数，并传入类似 Express 请求对象的参数
        // 包含请求方法、解析后的请求体和解析后的 URL 参数
        result = respond({
          method: type,
          body: JSON.parse(body),
          query: param2Obj(url)
        });
      } else {
        // 如果 respond 不是函数，则直接将其赋值给 result
        result = respond;
      }
      // 使用 Mock.js 对结果进行模拟处理
      return Mock.mock(result);
    };
  }

  // 遍历 mocks 数组中的每个模拟数据配置
  for (const i of mocks) {
    // 使用 Mock.js 的 mock 方法拦截匹配的请求
    // 第一个参数是一个正则表达式，用于匹配请求的 URL
    // 第二个参数是请求方法（默认为 get）
    // 第三个参数是经过转换后的响应处理函数
    Mock.mock(new RegExp(i.url), i.type || 'get', XHR2ExpressReqWrap(i.response));
  }
}

// 导出 mocks 数组和 mockXHR 函数，供其他模块使用
module.exports = {
  mocks,
  mockXHR
};