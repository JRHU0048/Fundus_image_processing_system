/**
 * 此函数的作用是将 URL 中的查询参数部分解析为一个对象。
 * @param {string} url - 包含查询参数的完整 URL 字符串。
 * @returns {Object} - 解析后的查询参数对象，键为参数名，值为参数值。
 */
function param2Obj(url) {
  // 先使用 split('?') 方法将 URL 按问号分割成数组，取索引为 1 的元素，即查询参数部分
  // 再使用 decodeURIComponent 对查询参数进行解码，避免出现编码后的特殊字符
  // 最后使用 replace(/\+/g, ' ') 将查询参数中的加号替换为空格，因为 URL 中加号通常代表空格
  const search = decodeURIComponent(url.split('?')[1]).replace(/\+/g, ' ');

  // 如果查询参数部分为空，直接返回一个空对象
  if (!search) {
      return {};
  }

  // 用于存储解析后的查询参数对象
  const obj = {};
  // 使用 split('&') 方法将查询参数按 & 符号分割成多个参数项数组
  const searchArr = search.split('&');

  // 遍历参数项数组
  searchArr.forEach(v => {
      // 查找当前参数项中 = 符号的索引位置
      const index = v.indexOf('=');
      // 如果找到了 = 符号
      if (index !== -1) {
          // 提取 = 符号前面的部分作为参数名
          const name = v.substring(0, index);
          // 提取 = 符号后面的部分作为参数值
          const val = v.substring(index + 1, v.length);
          // 将参数名和参数值作为键值对添加到 obj 对象中
          obj[name] = val;
      }
  });

  // 返回解析后的查询参数对象
  return obj;
}

// 导出 param2Obj 函数，以便其他模块可以使用
module.exports = {
  param2Obj
};

// 这个 param2Obj 函数的主要功能是将 URL 中的查询参数部分解析成一个 JavaScript 对象，方便后续对这些参数进行操作和使用